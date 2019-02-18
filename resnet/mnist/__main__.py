import click

from resnet.cifar10.train import train
from resnet.cifar10.train_ensemble import train_ensemble

@click.group()
@click.pass_context
def cli(ctx):
    if not hasattr(ctx, 'obj') or ctx.obj is None:
        setattr(ctx, 'obj', {})
    ctx.obj['dataset'] = 'mnist'


cli.add_command(train, name='train')
cli.add_command(train_ensemble, name='train_ensemble')


if __name__ == '__main__':
    cli()
