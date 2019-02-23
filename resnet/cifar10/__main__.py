import click

from resnet.cifar10.train import train
from resnet.cifar10.evaluate import evaluate
from resnet.cifar10.attack import attack
from resnet.cifar10.certify import certify
#from resnet.cifar10.infer import infer
#from resnet.cifar10.train_ensemble import train_ensemble


@click.group()
def cli():
    pass


cli.add_command(train, name='train')
cli.add_command(evaluate, name='evaluate')
cli.add_command(attack, name='attack')
cli.add_command(certify, name='certify')
#cli.add_command(train_ensemble, name='train_ensemble')
#cli.add_command(infer, name='infer')


if __name__ == '__main__':
    cli()
