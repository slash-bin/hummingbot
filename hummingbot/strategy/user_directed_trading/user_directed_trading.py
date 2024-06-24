import asyncio
import logging
import platform
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.market_order import MarketOrder
from hummingbot.remote_iface.messages import (
    ExchangeInfo,
    ExchangeInfoCommandMessage,
    OpenOrderInfo,
    UserDirectedCancelCommandMessage,
    UserDirectedListActiveOrdersCommandMessage,
    UserDirectedTradeCommandMessage,
)
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase

if TYPE_CHECKING:
    from hummingbot.client.hummingbot_application import HummingbotApplication  # noqa: F401

udt_logger: Optional[logging.Logger] = None


async def _custom_stop_loop(self: "HummingbotApplication", skip_order_cancellation: bool = False):
    """
    Overrides the default stop_loop() method in HummingbotApplication to allow for cancelling orders in user directed
    trade strategy before stopping.

    The user directed trading strategy uses a different method for managing exchange connectors, and thus requires a
    different stop_loop() method to handle the cancellation of orders.
    """
    from hummingbot.core.rate_oracle.rate_oracle import RateOracle
    self.logger().info("stop command initiated.")
    self.notify("\nWinding down...")

    # Restore App Nap on macOS.
    if platform.system() == "Darwin":
        import appnope
        appnope.nap()

    # Call before_stop() on the strategy.
    try:
        strategy: UserDirectedTradingStrategy = self.strategy
        await strategy.before_stop()

        if self.stratey_task is not None and not self.strategy_task.cancelled():
            self.strategy_task.cancel()

        if RateOracle.get_instance().started:
            RateOracle.get_instance().stop()

        if self.markets_recorder is not None:
            self.markets_recorder.stop()

        if self.kill_switch is not None:
            self.kill_switch.stop()

        self.strategy_task = None
        self.strategy = None
        self.market_pair = None
        self.clock = None
        self.markets_recorder = None
        self.market_trading_pairs_map.clear()
    finally:
        # Restore the original stop loop function
        self.stop_loop = self._original_stop_loop
        del self._original_stop_loop


class UserDirectedTradingStrategy(StrategyPyBase):
    """
    User directed trading strategy.

    This strategy allows users to direct trading actions via MQTT and thus Hummingbot AI chat interface.

    Since the user may specify any arbitrary exchange connector and trading pair to trade on, this stratey uses custom
    methods to manage exchange connectors and trading pairs. The exchange connectors are created on-the-fly and are
    not managed by the Hummingbot application.
    """
    _exchange_connectors_cache: Dict[Tuple[str, str], ExchangeBase]
    _is_stopping: bool

    @classmethod
    def logger(cls) -> logging.Logger:
        global udt_logger
        if udt_logger is None:
            udt_logger = logging.getLogger(__name__)
        return udt_logger

    def init_params(self):
        self._exchange_connectors_cache = {}
        self._is_stopping = False

    async def get_connected_exchanges(self) -> Dict[str, ExchangeBase]:
        """
        Get a dictionary of connected exchanges for fulfilling informational requests.

        The exchange connectors returned by this method are not meant for trading, but for informational purposes only.
        """
        from hummingbot.client.config.security import Security
        from hummingbot.client.hummingbot_application import HummingbotApplication
        from hummingbot.user.user_balances import UserBalances

        await Security.wait_til_decryption_done()

        app: HummingbotApplication = HummingbotApplication.main_application()
        user_balances: UserBalances = UserBalances.instance()
        return {
            exchange_name: user_balances.connect_market(
                exchange_name,
                app.client_config_map,
                **Security.api_keys(exchange_name)
            )
            for exchange_name in user_balances.all_available_balances_all_exchanges().keys()
        }

    async def get_trading_exchange_connector(self, exchange_name: str, trading_pair: str) -> ExchangeBase:
        """
        Get the exchange connector for the given exchange and trading pair for trading purpose.

        The exchange connectors created by the user directed trading strategy are managed by this strategy directly,
        instead of being managed by Hummingbot application. This is due to the assumption in Hummingbot application
        that each exchange connector corresponds to one exchange only.
        """
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ReadOnlyClientConfigAdapter, get_connector_class
        from hummingbot.client.config.security import Security
        from hummingbot.client.hummingbot_application import HummingbotApplication
        from hummingbot.client.settings import AllConnectorSettings, ConnectorSetting, ConnectorType
        from hummingbot.connector.exchange.paper_trade import create_paper_trade_market
        from hummingbot.user.user_balances import UserBalances

        await Security.wait_til_decryption_done()

        exchange_key: Tuple[str, str] = (exchange_name, trading_pair)
        if exchange_key not in self._exchange_connectors_cache:
            app: HummingbotApplication = HummingbotApplication.main_application()
            user_balances: UserBalances = UserBalances.instance()
            client_config_map: ClientConfigMap = app.client_config_map
            network_timeout: float = float(client_config_map.commands_timeout.other_commands_timeout)
            try:
                await asyncio.wait_for(
                    user_balances.update_exchanges(client_config_map=client_config_map, reconnect=True),
                    network_timeout
                )
            except asyncio.TimeoutError:
                app.notify("\nCould not fetch exchange information. Please check your network connection.")
                raise

            if exchange_name not in user_balances.all_available_balances_all_exchanges():
                raise ValueError(f"Exchange {exchange_name} is not connected and available.")

            conn_setting: ConnectorSetting = AllConnectorSettings.get_connector_settings()[exchange_name]
            if exchange_name.endswith("paper_trade") and conn_setting.type == ConnectorType.Exchange:
                connector: ExchangeBase = create_paper_trade_market(
                    conn_setting.parent_name,
                    client_config_map=client_config_map,
                    trading_pairs=[trading_pair]
                )
                paper_trade_account_balance: Dict[str, float] = (
                    client_config_map.paper_trade.paper_trade_account_balance
                )
                if paper_trade_account_balance is not None:
                    for asset, balance in paper_trade_account_balance.items():
                        connector.set_balance(asset, balance)
            else:
                api_keys: Dict[str, str] = Security.api_keys(exchange_name)
                read_only_config: ReadOnlyClientConfigAdapter = ReadOnlyClientConfigAdapter.lock_config(client_config_map)
                init_params: Dict[str, Any] = conn_setting.conn_init_parameters(
                    trading_pairs=[trading_pair],
                    trading_required=True,
                    api_keys=api_keys,
                    client_config_map=read_only_config
                )
                connector_class: Type[ExchangeBase] = get_connector_class(exchange_name)
                connector: ExchangeBase = connector_class(**init_params)
            self._exchange_connectors_cache[exchange_key] = connector

        return self._exchange_connectors_cache[exchange_key]

    async def create_market_trading_pair_tuple(self, exchange: ExchangeBase, trading_pair: str) -> MarketTradingPairTuple:
        base_asset, quote_asset = trading_pair.split("-")
        exchange_connector: ExchangeBase = await self.get_trading_exchange_connector(exchange, trading_pair)
        return MarketTradingPairTuple(
            market=exchange_connector,
            trading_pair=trading_pair,
            base_asset=base_asset,
            quote_asset=quote_asset
        )

    async def before_stop(self):
        # Cancel all open orders, and mark the strategy as stopped
        from hummingbot.client.hummingbot_application import HummingbotApplication
        app: HummingbotApplication = HummingbotApplication.main_application()
        self._is_stopping = True
        app.notify("\nCancelling all open orders...")
        for exchange_connector in self._exchange_connectors_cache.values():
            try:
                await exchange_connector.cancel_all(HummingbotApplication.KILL_TIMEOUT)
            except Exception:
                self.logger().error("Error canceling outstanding orders.", exc_info=True)
                pass
        # Give some time for cancellation events to trigger
        await asyncio.sleep(1)
        # Clean up all exchange connectors
        self._exchange_connectors_cache.clear()

    def start(self, clock: Clock, timestamp: float):
        # Overwrite the `stop_loop()` method in `HummingbotApplication` to allow for cancelling orders before
        # stopping the strategy
        global _custom_stop_loop
        from hummingbot.client.hummingbot_application import HummingbotApplication
        app: HummingbotApplication = HummingbotApplication.main_application()
        app._original_stop_loop = app.stop_loop
        app.stop_loop = _custom_stop_loop

    def tick(self, timestamp: float):
        # Start and tick all exchange connectors
        for exchange_connector in self._exchange_connectors_cache.values():
            if exchange_connector.clock is None:
                exchange_connector.start(self.clock, timestamp)
            exchange_connector.tick(timestamp)

    def format_status(self) -> str:
        # Implement your format_status logic here
        pass

    async def mqtt_exchange_info(self, exchange: Optional[str]) -> ExchangeInfoCommandMessage.Response:
        exchanges: Dict[str, ExchangeBase] = await self.get_connected_exchanges()
        for exchange_connector in exchanges.values():
            await exchange_connector._update_balances()
        exchange_info: List[ExchangeInfo] = [
            ExchangeInfo(
                name=exchange_name,
                trading_pairs=await exchange_connector.all_trading_pairs(),
                balances={
                    asset: float(balance)
                    for asset, balance in exchange_connector.get_all_balances().items()
                }
            )
            for exchange_name, exchange_connector in exchanges.items()
        ]
        return ExchangeInfoCommandMessage.Response(exchanges=exchange_info)

    async def mqtt_list_active_orders(
            self,
            exchange: Optional[str],
            trading_pair: Optional[str]
    ) -> UserDirectedListActiveOrdersCommandMessage.Response:
        tracked_limit_orders: List[Tuple[ConnectorBase, LimitOrder]] = self.order_tracker.tracked_limit_orders
        tracked_market_orders: List[Tuple[ConnectorBase, MarketOrder]] = self.order_tracker.tracked_market_orders
        open_orders: List[OpenOrderInfo] = []

        for connector, limit_order in tracked_limit_orders:
            if exchange is not None and connector.name != exchange:
                continue
            if trading_pair is not None and limit_order.trading_pair != trading_pair:
                continue
            open_orders.append(OpenOrderInfo(
                exchange=connector.name,
                trading_pair=limit_order.trading_pair,
                order_id=limit_order.client_order_id,
                order_type="LIMIT",
                price=limit_order.price,
                amount=limit_order.quantity,
                status=limit_order.status
            ))

        for connector, market_order in tracked_market_orders:
            if exchange is not None and connector.name != exchange:
                continue
            if trading_pair is not None and market_order.trading_pair != trading_pair:
                continue
            open_orders.append(OpenOrderInfo(
                exchange=connector.name,
                trading_pair=market_order.trading_pair,
                order_id=market_order.client_order_id,
                order_type="MARKET",
                price=None,
                amount=market_order.quantity,
                status=market_order.status
            ))

        return UserDirectedTradeCommandMessage.Response(active_orders=open_orders)

    async def mqtt_user_directed_trade(
            self,
            exchange: str,
            trading_pair: str,
            is_buy: bool,
            is_limit_order: bool,
            limit_price: Optional[float],
            amount: float,
    ) -> UserDirectedTradeCommandMessage.Response:
        if is_limit_order and limit_price is None:
            raise ValueError("Limit price must not be None for a limit order")

        trading_pair_tuple: MarketTradingPairTuple = await self.create_market_trading_pair_tuple(exchange, trading_pair)
        if is_limit_order:
            if is_buy:
                order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=trading_pair_tuple,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=Decimal(f"{limit_price:g}")
                )
            else:
                order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=trading_pair_tuple,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=Decimal(f"{limit_price:g}")
                )
        else:
            if is_buy:
                order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=trading_pair_tuple,
                    amount=amount
                )
            else:
                order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=trading_pair_tuple,
                    amount=amount
                )

        return UserDirectedTradeCommandMessage.Response(order_id=order_id)

    async def mqtt_user_directed_cancel(
            self,
            exchange: str,
            trading_pair: str,
            order_id: str
    ) -> UserDirectedCancelCommandMessage.Response:
        trading_pair_tuple: MarketTradingPairTuple = await self.create_market_trading_pair_tuple(exchange, trading_pair)
        self.cancel_order(trading_pair_tuple, order_id)
        return UserDirectedCancelCommandMessage.Response(order_id=order_id)
