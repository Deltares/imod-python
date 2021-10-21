import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"), base_url="https://gitlab.com/deltares/imod/imod-python"
)
