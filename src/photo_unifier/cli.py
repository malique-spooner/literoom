import logging
import click

# Simple, visible logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("photo_unifier")

@click.group()
def app():
    """photo-unifier CLI (MVP)"""
    pass

@app.command()
@click.option("--who", default="world", help="Who to greet")
def hello(who: str):
    """Quick sanity check command"""
    log.info("Hello, %s! CLI is working.", who)

if __name__ == "__main__":
    app()
