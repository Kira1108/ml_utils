from ml_utils.cv import generate_image_router
import os


import click

@click.command()
@click.option('-p', default=".", help='path to generate router file')
@click.option('-fn', default = "image_api.py",help='filename of the router')
def cli(p, fn):
    generate_image_router(p, fn)
    click.echo(f"Router generated successfully {os.path.join(p, fn)}")