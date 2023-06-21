from typing import *

from reactpy import component, hooks, html
from reactpy.core.types import VdomChildren


@component
def TabBar(children: VdomChildren):
    return html.div(
        {
            "class": "tabs tabs-boxed",
        },
        children,
    )


@component
def TabBarItem(
    children: VdomChildren, index: int, current_tab: int, set_current_tab: Callable
):
    selected = hooks.use_memo(lambda: index == current_tab, [current_tab])

    def on_click(_):
        set_current_tab(index)

    return html.button(
        {
            "class": "tab " + ("tab-active" if selected else ""),
            "role": "tab",
            "onclick": on_click,
        },
        children,
    )


@component
def TabPanel(
    children: VdomChildren, index: int, current_tab: int, set_current_tab: Callable
):
    selected = hooks.use_memo(lambda: index == current_tab, [current_tab])

    return html.div(
        {
            "class": "tab-content " + ("active" if selected else "hidden"),
        },
        children,
    )


@component
def Tabs(tabs, props={}):
    current_tab, set_current_tab = hooks.use_state(0)

    return html.div(
        props,
        TabBar(
            html._(
                *[
                    TabBarItem(tab.title(), i, current_tab, set_current_tab)
                    for i, tab in enumerate(tabs)
                ]
            )
        ),
        *[
            TabPanel(tab.ui(), i, current_tab, set_current_tab)
            for i, tab in enumerate(tabs)
        ],
    )
