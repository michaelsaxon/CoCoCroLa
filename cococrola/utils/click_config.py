import click
import yaml
import os

# simple code for storing defaults in local yaml files that don't get sync'd
# use this to write your defaults if you're running the same commands over and over

# usage:
# @click.command(cls=CommandWithConfigFile('config_file'))
def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                if os.path.exists(config_file):
                    with open(config_file) as f:
                        config_data = yaml.safe_load(f)
                        for param, value in ctx.params.items():
                            if value is None and param in config_data:
                                ctx.params[param] = config_data[param]

            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass