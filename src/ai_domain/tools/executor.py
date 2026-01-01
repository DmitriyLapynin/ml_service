class ToolExecutor:
    def __init__(self, registry, telemetry):
        self.registry = registry
        self.telemetry = telemetry

    async def execute(self, *, tool_name: str, args: dict, state):
        tool = self.registry.get(tool_name)

        if not tool:
            raise ValueError(f"Tool {tool_name} not registered")

        self.telemetry.event(
            "tool_call",
            {
                "trace_id": state.trace_id,
                "tool": tool_name,
            },
        )

        return await tool.run(
            args=args,
            credentials=state.credentials,
            trace_id=state.trace_id,
        )