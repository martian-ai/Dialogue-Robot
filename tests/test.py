from tools.tool_llm.online_wenxin.llm_clients import OpenAiClient

openai_client = OpenAiClient()

resp = openai_client.request_single_query("你好")

print(resp)