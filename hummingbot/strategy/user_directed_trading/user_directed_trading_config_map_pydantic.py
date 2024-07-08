from pydantic import Field

from hummingbot.client.config.strategy_config_data_types import BaseStrategyConfigMap


class UserDirectedTradingConfigMap(BaseStrategyConfigMap):
    strategy: str = Field(default="user_directed_trading", client_data=None)
