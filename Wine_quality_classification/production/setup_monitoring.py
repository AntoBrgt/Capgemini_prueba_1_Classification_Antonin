import os

def setup_prometheus():
    """
    Run prometheus with docker for the monitoring.
    Open it on http://localhost:9090
    """

    os.system("docker run --name prometheus -p 9090:9090 prom/prometheus")

    prometheus_config = """
    global:
      scrape_interval:     15s 
      evaluation_interval: 15s 

    scrape_configs:
      - job_name: 'flask_app'
        static_configs:
          - targets: ['127.0.0.1:5000']
    """
    with open('prometheus.yml', 'w') as f:
        f.write(prometheus_config)

    os.system("docker cp prometheus.yml prometheus:/etc/prometheus/")
    os.system("docker restart prometheus")

if __name__ == "__main__":
    setup_prometheus()
