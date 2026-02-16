######################################################################
# Copyright (c) 2025 Luca Prestini
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################

from __future__ import annotations

from flask import Flask, jsonify, send_from_directory, request, abort
import json
from pathlib import Path
from typing import Any, Dict, Optional
import socket
import logging
import ipaddress
import os


class MLRunnerServer:
    """
    Small Flask server that:
      - serves index.html at /
      - serves JSON contents at /api/data

    Usage:
        server = JsonDashboardServer(
            data_path="somepath.json",
            web_dir="./web",  # folder that contains index.html
        )
        server.run(host="0.0.0.0", port=8000)

    Or threaded:
        server.start_threaded(host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        data_path: str | Path,
        web_dir: str | Path = "./",
        host: str = "0.0.0.0",
        port: int = 8000,
        ip_allow_list=[],
        use_ip_allow_list=False,
    ):
        """
        Args:
        data_path: Path to where the data json is
        web_dir: Path to the index.thml file
        host: interface to listen to - 0.0.0.0 means any machine that can ping your computer can access this - if they are on the network and your firewall allows it. *NOTE*
        Be aware that opening a port through a firewall on a network that is not set up correctly may be dangerous without allow list or other restrictions in place
        port: what port to use to access the server
        ip_allow_list: This should be a list of strings representing the subnetworks you want to restrict access to. If this is not enalbed and the network you run this server on is public - it can be dangerous.
        e.g ["100.123.122.0/24","100.124.124.0/24"]
        use_ip_allow_list: Wether to use the IP allow list
        """
        self.data_path = Path(data_path)
        self.web_dir = Path(web_dir).resolve()
        self.url_prefix = "".rstrip("/")  # allow "" or "/dash"

        self.app = Flask("MLRunner dashboard")
        self.app.config["JSON_SORT_KEYS"] = False
        self.app.logger.setLevel(logging.INFO)
        try:
            self.app.json.sort_keys = False
        except Exception:
            pass

        if use_ip_allow_list:
            self.allowed_nets = [ipaddress.ip_network(i) for i in ip_allow_list]
            self.app.before_request(self._restrict_network)
        self._register_routes()

    def _restrict_network(self):
        """If enabled allow access only to people on specific subnetworks"""
        ip = ipaddress.ip_address(request.remote_addr)
        if not any(ip in net for net in self.allowed_nets):
            abort(403)

    def get_machine_ip(
        self,
    ):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No traffic is actually sent
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()

    def log_web_address_once(self):
        _ip = self.get_machine_ip()
        address = f"http://{_ip}:{self.port}"
        self.app.logger.info(f"MLRunner Dashboard is running at: {address}")

    def inform_about_server(self):
        self.app.logger.warning(
            "\n\nHELLO! PLEASE READ THE FOLLOWING:\n\n"
            "If you are running the server on a network and you need other people to access the dashboard to see whats going on, make sure that the network is set up correctly!\n"
            "Opening a port through a firewall on the wrong network may expose the server to the public. Causing a potential VULNERABILITY. If you are not sure what you are doing you can:\n"
            "    - Speak with your system adminstrator or IT to ask which port is free and not set to public on the network.\n"
            "    - Do not open a port on your firewall\n"
            "    - Set up the IP allowlist to allow only certain subnets to access this service\n"
            "    - Disable the web server funcionality by rerunning the app with --no-web_server False\n"
            "Thank you for reading!\n\n"
        )

    def _read_json_file(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            return {"error": "json file not found", "path": str(self.data_path)}

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                ids = data.get("job_id", [])
                sts = data.get("status", [])
                timestamp = data.get("timestamp", [])
                start_time = data.get("started_at", [])
                error_log = data.get("error_log", [])
                model2run = data.get("model2run", [])
                n = max(len(ids), len(sts))
                n = n if n != 0 else 1
                data = [
                    {
                        "ID": ids[i] if i < len(ids) else "",
                        "Model name": model2run[i] if i < len(model2run) else "",
                        "Status": sts[i] if i < len(sts) else "",
                        "Submission Time": timestamp[i] if i < len(sts) else "",
                        "Start Time": start_time[i] if i < len(sts) else "",
                        "Error log": error_log[i] if i < len(sts) else "",
                    }
                    for i in range(n)
                ]
                return data

        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON", "details": str(e)}
        except Exception as e:
            return {"error": "Failed to read JSON", "details": str(e)}

    def _register_routes(self) -> None:
        # Serve the UI
        @self.app.get(f"{self.url_prefix}/")
        def index():
            return send_from_directory(self.web_dir, "index.html")

        # Serve the data
        @self.app.get(f"{self.url_prefix}/api/data")
        def api_data():
            return jsonify(self._read_json_file())

        # Optional: serve other static assets if you add them later (css/js)
        @self.app.get(f"{self.url_prefix}/<path:filename>")
        def static_files(filename: str):
            return send_from_directory(self.web_dir, filename)

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False) -> None:
        self.app.run(host=host, port=port, debug=debug)

    def start_threaded(
        self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False
    ):
        """
        Start Flask in a background thread (useful if you have a main loop).
        Note: Flask dev server isn't for production, but fine for LAN dashboards.
        """
        import threading

        self.host = host
        self.port = port
        self.inform_about_server()
        self.log_web_address_once()

        t = threading.Thread(
            target=self.run,
            kwargs={"host": self.host, "port": self.port, "debug": debug},
            daemon=True,
        )
        t.start()
        return t
