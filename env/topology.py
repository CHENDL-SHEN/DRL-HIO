import yaml
import networkx as nx
import math

from typing import Dict, List, Tuple

class SCCTopology:

    def __init__(self, yaml_path: str):

        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.graph = nx.DiGraph()

        self.tracks = {
            t['id']: t
            for t in self.config['tracks']
        }

        self.vehicles = {
            v['id']: v
            for v in self.config['vehicles']
        }

        self.workstations = {
            w['id']: w
            for w in self.config['workstations']
        }

        self._build_graph()

    def _build_graph(self):

        for ws_id, ws_data in self.workstations.items():

            self.graph.add_node(
                ws_id,
                pos=ws_data['pos'],
                type=ws_data['type'],
                connected_tracks=ws_data['connected_tracks']
            )

        node_ids = list(self.graph.nodes())

        for i in range(len(node_ids)):

            for j in range(i + 1, len(node_ids)):

                node_a = node_ids[i]
                node_b = node_ids[j]

                tracks_a = set(
                    self.graph.nodes[node_a]['connected_tracks']
                )

                tracks_b = set(
                    self.graph.nodes[node_b]['connected_tracks']
                )

                common_tracks = tracks_a.intersection(tracks_b)

                if common_tracks:

                    pos_a = self.graph.nodes[node_a]['pos']
                    pos_b = self.graph.nodes[node_b]['pos']

                    dist = math.dist(pos_a, pos_b)

                    for track in common_tracks:

                        self.graph.add_edge(
                            node_a,
                            node_b,
                            weight=dist,
                            track=track
                        )

                        self.graph.add_edge(
                            node_b,
                            node_a,
                            weight=dist,
                            track=track
                        )

    def get_distance(
        self,
        node_a: str,
        node_b: str
    ) -> float:

        if self.graph.has_edge(node_a, node_b):

            return self.graph[node_a][node_b]['weight']

        return float('inf')