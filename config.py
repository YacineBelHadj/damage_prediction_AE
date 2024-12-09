
from dynaconf import Dynaconf
import pyrootutils
import os 


# combines find_root() and set_root() into one method
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)
# add root to os environmental variables
os.environ["HOME"] = root.absolute().as_posix()
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.toml', 'config/.secrets.toml','config/structuralstate.toml'],
    root_path=root
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
settings.set('root', root)

if __name__=='__main__':
    print(settings.to_dict())