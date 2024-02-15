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


# inplace
def init_function_dict(di):
    empty_dict = {"name": "", "description": ""}
    for k, v in empty_dict.items():
        di[k] = v


def parse_methods(docs, methods_start_idx, methods_stop_idx):
    functions_df = pd.DataFrame(columns=["name", "description"])

    function_dict = {}
    init_function_dict(function_dict)

    for child in docs.contents[methods_start_idx:methods_stop_idx]:
        if child.name == TAG_NAMES["function_name"]:
            if function_dict["name"] != "":
                functions_df.loc[len(functions_df)] = function_dict
                init_function_dict(function_dict)

            function_name = str(child.contents[0])
            function_dict["name"] = function_name

        elif child.name == "p":
            if child.contents[0] in REDUNDANT_NOTES:
                continue
            function_dict[f"description"] += child.get_text() + " "

        elif child.name == "ul":
            function_dict[f"description"] += parse_list(child) + " "

        elif child.name == "h4":
            function_dict[f"description"] += (
                child.get_text().replace("\u200b", "") + " "
            )

        elif child.name == "table":
            function_dict[f"description"] += parse_table(child) + " "

        else:
            with open("other_tags.html", "a") as f:
                f.write("IN FUNCTIONS:\n")
                f.write(child.prettify())
    functions_df.loc[len(functions_df)] = function_dict

    functions_df.to_csv("functions.csv", index=False)


def parse_component(docs, component_start_idx, component_stop_idx):
    component_df = pd.DataFrame(columns=["name", "type", "description"])
    component_dict = {
        "name": "",
        "type": "",
        "description": "",
    }
    for child in docs.contents[component_start_idx:component_stop_idx]:
        if child.name == "h1":
            component_dict["name"] = child.get_text()

        elif child.name == "p":
            component_dict["description"] += child.get_text() + " "

            if "service" in component_dict["description"]:
                component_dict["type"] = "service"
            else:
                component_dict["type"] = "component"

        elif child.name == "ul":
            component_dict["description"] += parse_list(child) + " "

        else:
            with open("other_tags.html", "a") as f:
                f.write("IN COMPONENT:\n")
                f.write(child.prettify())
    component_df.loc[len(component_df)] = component_dict
    component_df.to_csv("component.csv", index=False)


with open("index_cover.html") as fp:
    html_doc = fp.read()

soup = BeautifulSoup(html_doc, "html.parser")

docs = soup.find_all("div", class_="theme-doc-markdown markdown")[0]

REDUNDANT_NOTES = [
    "Attributes in the result:",
    "Attributes in the result (only the ones available are shown):",
    "Parameters:",
]
TAG_NAMES = {"function_name": "h3"}

component_start_idx = 0
component_stop_idx = docs.contents.index(docs.find("h2"))
methods_start_idx = docs.contents.index(docs.find("h2")) + 1
methods_stop_idx = docs.contents.index(docs.select('h2[id*="http-endpoint-"]')[0])

parse_component(docs, component_start_idx, component_stop_idx)
parse_methods(docs, methods_start_idx, methods_stop_idx)
