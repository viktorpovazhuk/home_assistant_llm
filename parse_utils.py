from bs4 import BeautifulSoup
import pandas as pd


def parse_table(table):
    properties_list = []
    for row in table.css.select("table > tbody > tr"):
        col_values = []
        for child in row.css.select("tr > td"):
            val = child.find("p").get_text()
            col_values.append(val)
        properties_list.append(
            {"name": col_values[0], "type": col_values[1], "description": col_values[2]}
        )
    return str(properties_list)


def parse_list(list_el):
    str_items = []
    for li in list_el.css.select("ul > li:not(li li)"):
        item = ""
        subitems = []
        for ch in li.children:
            if ch.name != "ul":
                item += ch.get_text()
            else:
                subitems.append(ch.get_text())
        item += ".".join(subitems) + "."
        str_items.append(item)
    str_items = " ".join(str_items)
    return str_items


def parse_div(div_el):
    return div_el.get_text(separator=". ")


# inplace
def init_function_dict(di):
    empty_dict = {"name": "", "description": ""}
    for k, v in empty_dict.items():
        di[k] = v
