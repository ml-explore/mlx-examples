run:
	mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --trust-remote-code --port 8722

k:
	ps -ef|grep 'mlx_lm.server'|awk '{print $2}'|xargs kill -9

w:
	curl -X GET "http://127.0.0.1:9000/api/ai/WriteBlogRandomlyWithLLM?model=MLXLMServer" -H  "Request-Origion:SwaggerBootstrapUi" -H  "accept:*/*"

c:
	conda activate m3mlx