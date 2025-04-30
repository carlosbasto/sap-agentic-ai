
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, function, description, parameters):
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }

    def get_description_for_prompt(self):
        return {
            name: {
                "description": entry["description"],
                "parameters": entry["parameters"]
            } for name, entry in self.tools.items()
        }

    def get_callable(self, name):
        return self.tools.get(name, {}).get("function")