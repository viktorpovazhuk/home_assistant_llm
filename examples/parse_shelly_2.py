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
    return properties_list


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
        item += " " + ".".join(subitems) + "."
        str_items.append(item)
    str_items = " ".join(str_items)
    return str_items


# inplace
def init_function_dict(di):
    empty_dict = {
        "function_name": "",
        "description": "",
        "request_properties": None,
        "request_notes": "",
        "response_properties": None,
        "response_notes": "",
    }
    for k, v in empty_dict.items():
        di[k] = v


def parse_methods(docs, methods_start_idx, methods_stop_idx):
    functions_df = pd.DataFrame(
        columns=[
            "function_name",
            "description",
            "request_properties",
            "request_notes",
            "response_properties",
            "response_notes",
        ]
    )

    frame_name = None
    function_dict = {}
    init_function_dict(function_dict)

    for child in docs.contents[methods_start_idx:methods_stop_idx]:
        if child.name == "h2":
            if function_dict["function_name"] != "":
                functions_df.loc[len(functions_df)] = function_dict
                init_function_dict(function_dict)
                frame_name = None
            function_name = str(child.contents[0])
            function_dict["function_name"] = function_name

        elif child.name == "p" and frame_name is None:
            function_dict[f"description"] += child.get_text()

        elif child.name == "h4":
            frame_name = str(child.contents[0]).lower()

        elif child.name == "p":
            if child.contents[0] in redundant_notes:
                continue
            function_dict[f"{frame_name}_notes"] += child.get_text()

        elif child.name == "ul":
            function_dict[f"{frame_name}_notes"] += parse_list(child)

        elif child.name == "table":
            function_dict[f"{frame_name}_properties"] = parse_table(child)

        else:
            with open("other_tags.html", "a") as f:
                f.write("IN FUNCTIONS:\n")
                f.write(child.prettify())

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
            component_dict["description"] += child.get_text()
            if "service" in component_dict["description"]:
                component_dict["type"] = "service"
            else:
                component_dict["type"] = "component"
            component_df.loc[len(component_df)] = component_dict
        else:
            with open("other_tags.html", "a") as f:
                f.write("IN COMPONENT:\n")
                f.write(child.prettify())
    component_df.to_csv("component.csv", index=False)


with open("index_shelly.html") as fp:
    html_doc = fp.read()

soup = BeautifulSoup(html_doc, "html.parser")

docs = soup.find_all("div", class_="theme-doc-markdown markdown")[0]

redundant_notes = [
    "Attributes in the result:",
    "Attributes in the result (only the ones available are shown):",
    "Parameters:",
]

component_start_idx = 0
component_stop_idx = docs.contents.index(docs.find("h2"))
methods_start_idx = docs.contents.index(docs.find("h2"))
methods_stop_idx = docs.contents.index(docs.select('h2[id*="http-endpoint-"]')[0])

parse_component(docs, component_start_idx, component_stop_idx)
parse_methods(docs, methods_start_idx, methods_stop_idx)
