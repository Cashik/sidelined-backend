from src import schemas, enums

project_tokens = {
    "RIZ": schemas.Token(
        chain_id=enums.ChainID.BASE,
        address="0x67543CF0304C19CA62AC95ba82FD4F4B40788dc1",
        interface=enums.TokenInterface.ERC20,
        decimals=8,
        symbol="RIZ",
        name="Rivals Network"
    ),
    "wRIZ": schemas.Token(
        chain_id=enums.ChainID.BASE,
        address="0xA70acF9Cbb8CA5F6c2A9273283fb17C195ab7a43",
        interface=enums.TokenInterface.ERC20,
        decimals=8,
        symbol="wRIZ",
        name="Staked RIZ"
    ),
    "ZNL": schemas.Token(
        chain_id=enums.ChainID.ARBITRUM,
        address="0x78bDE7b6C7eB8f5F1641658c698fD3BC49738367",
        interface=enums.TokenInterface.ERC721,
        decimals=0,
        symbol="ZNL",
        name="ZNode License"
    )
}


pro_plan_requirements = [
    schemas.TokenRequirement(
        token=project_tokens["RIZ"],
        balance=20_000,
        buy_link="https://aerodrome.finance/swap?from=eth&to=0x67543cf0304c19ca62ac95ba82fd4f4b40788dc1&chain0=8453&chain1=8453"
    ),
    schemas.TokenRequirement(
        token=project_tokens["wRIZ"],
        balance=15_000,
        buy_link="https://stake.rivalz.ai"
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.ETHEREUM,
            address="0xc555D625828c4527d477e595fF1Dd5801B4a600e",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="MON",
            name="Monprotocol"
        ),
        balance=200
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0xBC7F9Fc7693AB20aDbF913537Ecb6535864C6c5C",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="aTRUST",
            name="Staked $TRUST"
        ),
        balance=10_000
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0xc841b4ead3f70be99472ffdb88e5c3c7af6a481a",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="TRUST",
            name="Trust me bros"
        ),
        balance=10_000
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0x79dacb99a8698052a9898e81fdf883c29efb93cb",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="ACOLYT",
            name="SIgnal"
        ),
        balance=1_000
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0x23cdabCDD4a246758A9F00F8ce60FF33B5e858C9",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="DATDAO",
            name="by TRUST"
        ),
        balance=100_000
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0x548D3B444da39686d1a6F1544781d154e7cD1EF7",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="sKAITO",
            name="Staked Kaito"
        ),
        balance=100
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0x98d0baa52b2D063E780DE12F615f963Fe8537553",
            interface=enums.TokenInterface.ERC20,
            decimals=18,
            symbol="KAITO",
            name="Kaito"
        ),
        balance=100
    ),
]

ultra_plan_requirements = [
    schemas.TokenRequirement(
        token=project_tokens["RIZ"],
        balance=150_000,
        buy_link="https://aerodrome.finance/swap?from=eth&to=0x67543cf0304c19ca62ac95ba82fd4f4b40788dc1&chain0=8453&chain1=8453"
    ),
    schemas.TokenRequirement(
        token=project_tokens["wRIZ"],
        balance=100_000,
        buy_link="https://stake.rivalz.ai"
    ),
    schemas.TokenRequirement(
        token=project_tokens["ZNL"],
        balance=1
    )
]


all_ai_models_ids = list(enums.Model)
basic_ai_models_ids = [enums.Model.GPT_4O, enums.Model.GPT_O4_MINI, enums.Model.GEMINI_2_5_FLASH]
pro_ai_models_ids = [enums.Model.GPT_4_1, enums.Model.GEMINI_2_5_PRO]


all_toolboxes_ids = list(enums.ToolboxList)
basic_toolboxes_ids = [enums.ToolboxList.BASIC]

# todo: явно разделять требования на наши и партнеров
# todo: на старте нужно проверять, можем ли мы проверять требования - все ли настройки валидные
subscription_plans = [
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.BASIC,
        name="Basic",
        requirements=[],
        max_credits=30,
        available_models_ids=basic_ai_models_ids,
        available_toolboxes_ids=basic_toolboxes_ids
    ),
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.PRO,
        name="Pro",
        requirements=pro_plan_requirements,
        max_credits=100,
        available_models_ids=pro_ai_models_ids+basic_ai_models_ids,
        available_toolboxes_ids=all_toolboxes_ids
    ),
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.ULTRA,
        name="Ultra",
        requirements=ultra_plan_requirements,
        max_credits=10000,
        available_models_ids=all_ai_models_ids,
        available_toolboxes_ids=all_toolboxes_ids
    )
]

def get_subscription_plan(subscription_id: enums.SubscriptionPlanType) -> schemas.SubscriptionPlanExtended:
    for plan in subscription_plans:
        if plan.id == subscription_id:
            return plan
    raise ValueError(f"Subscription plan with id {subscription_id} not found")


