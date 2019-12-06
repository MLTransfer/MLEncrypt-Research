import click


@click.group()
def cli():
    pass


@cli.command()
def send():
    print('send')


@cli.command()
def receive():
    print('receive')


if __name__ == '__main__':
    cli()
