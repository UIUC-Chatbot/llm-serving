# Nginx usage 

Start NGINX (no sudo) with custom config
```bash
# test it
nginx -t -c /home/kastanday/llm_serving/llm-server/nginx/nginx.conf

# start
nginx -c /home/kastanday/llm_serving/llm-server/nginx/nginx.conf
```

## Reload after changes
```bash
# note cat -pp is required if using `bat` instead of regular cat
kill -HUP $(cat -pp /home/kastanday/llm_serving/llm-server/nginx/nginx.pid)
```
Here, `$(cat /home/kastanday/llm_serving/llm-server/nginx/nginx.pid)` fetches the PID from your custom PID file, and `kill -HUP` sends the HUP (Hang UP) signal to the process, instructing it to reload its configuration.