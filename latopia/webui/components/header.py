from reactpy import hooks, html

TABS = {
    "ViTs": "vits",
    "RVC": "rvc",
    "DDSP": "ddsp",
}


def Header():
    active, set_active = hooks.use_state("vits")
    return html.div(
        html.h1({"class": "text-4xl mb-2"}, "Latopia"),
        html.div(
            {"class": "tabs tabs-boxed"},
            *[
                html.div(
                    {
                        "class": "tab tab-active" if active == id else "tab",
                        "on_click": lambda _: set_active(id),
                    },
                    name,
                )
                for name, id in TABS.items()
            ]
        ),
    )
