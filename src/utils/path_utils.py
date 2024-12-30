import os

def get_path_by_cluster_name(param, cfg):
    cluster_name = os.environ.get("CLUSTER_NAME", cfg.cluster_name)
    if param is None:
        return None
    elif cluster_name in param:
        return param[cluster_name]
    elif "default" in param:
        return param["default"]
    else:
        raise ValueError(f"Cluster name {cluster_name} not found in {param}.")