from os import makedirs


def get_params() -> dict:
    """
    Documentation
    """
    params = {
        "path_data": "base_datos_covid",
        "dataset": "BASE_SISVER_101121_EPI.csv"
    }
    return params


def mkdir(path: str) -> None:
    """
    Documentation
    """
    makedirs(
        path,
        exist_ok=True
    )
