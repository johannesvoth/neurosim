from graph_model import GraphModel
from pygame_view import PygameGraphApp


def main() -> None:
    model = GraphModel()
    app = PygameGraphApp(model)
    app.run()


if __name__ == "__main__":
    main()


