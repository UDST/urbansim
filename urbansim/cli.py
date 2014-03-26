import sys

from cliff.app import App
from cliff.commandmanager import CommandManager


class UrbansimCLI(App):
    def __init__(self):
        super(UrbansimCLI, self).__init__(
            description='UrbanSim Interface',
            version='0.2dev',
            command_manager=CommandManager('urbansim.commands'))


def main(args=sys.argv[1:]):
    app = UrbansimCLI()
    return app.run(args)
