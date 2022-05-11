
def class2label(pred):
    map_dict = {
        0: "['amendments']",
        1: "['counterparts']",
        2: "['governing laws']",
        3: "['government regulations']",
        4: "['terminations']",
        5: "['trade relations']",
        6: "['trading activities']",
        7: "['valid issuances']",
        8: "['waivers']",
        9: "['warranties']"
    }
    return map_dict[pred]