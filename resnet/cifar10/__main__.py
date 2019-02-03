import click

from resnet.cifar10.train import train
from resnet.cifar10.infer import infer
from resnet.cifar10.train_ensemble import train_ensemble


@click.group()
def cli():
    pass


cli.add_command(train, name='train')
cli.add_command(train, name='train_ensemble')
cli.add_command(infer, name='infer')


if __name__ == '__main__':
    cli()
