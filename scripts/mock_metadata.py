#!/usr/bin/env python3
"""Local GCE metadata proxy for isolated single-host/sub-host TPU v7x replicas on GKE.

Intercepts tpu-env and multi-host TPU worker metadata attributes that either
404 on GKE (triggering libtpu 30x retry loops) or expose the 4-host slice topology
when launching independent intra-host/intra-chip SGLang/TorchTPU replicas.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


class MockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def forward_to_real_metadata(self):
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        response_body = None
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CONSUMER_PROJECT_ID": "tpu-prod-env-one-vm",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": "1,1,1",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "tpu7x-8",
            "WRAP": "false,false,false",
            "ZONE": "us-central1-c",
        }
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            response_body = (
                "TPU_WORKER_ID: '0'\n"
                "ALT: 'false'\n"
                "CHIPS_PER_HOST_BOUNDS: '2,2,1'\n"
                "CONSUMER_PROJECT_ID: 'tpu-prod-env-one-vm'\n"
                "ENABLE_ICI_RESILIENCY: 'false'\n"
                "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\n"
                "HOST_BOUNDS: '1,1,1'\n"
                "INJECT_SLICE_BUILDER_FAULT: ''\n"
                "NODE_ID: '0'\n"
                "TOPOLOGY: '2x2x1'\n"
                "TPU_ACCELERATOR_TYPE: 'tpu7x-8'\n"
                "WRAP: 'false,false,false'\n"
                "ZONE: 'us-central1-c'\n"
            ).encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints") or path.endswith(
            "/computeMetadata/v1/instance/attributes/instance-id"
        ):
            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/hostname",
                headers={"Metadata-Flavor": "Google"},
            )
            hn = urllib.request.urlopen(req, timeout=2).read().decode().strip().split(".")[0]
            response_body = hn.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = b"tpu7x-8"
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"tpu-prod-env-one-vm"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


def run(server_class=ThreadingHTTPServer, handler_class=MockMetadataHandler, port=80):
    httpd = server_class(("127.0.0.1", port), handler_class)
    print(f"[mock_metadata] Serving on 127.0.0.1:{port}", file=sys.stderr, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    run()
