class RouterNode:
    async def __call__(self, state):
        # channel уже определён orchestrator'ом
        state.route = state.channel
        return state