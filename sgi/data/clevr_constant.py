synonyms = {
  "thing": ["thing", "object"],
  "sphere": ["sphere", "ball"],
  "cube": ["cube", "block"],
  "large": ["large", "big"],
  "small": ["small", "tiny"],
  "metal": ["metallic", "metal", "shiny"],
  "rubber": ["rubber", "matte"],
  "left": ["left of", "to the left of", "on the left side of"],
  "right": ["right of", "to the right of", "on the right side of"],
  "behind": ["behind"],
  "front": ["in_front_of"],
  "above": ["above"],
  "below": ["below"],
}

syn_attrs = {
  "thing": ["thing", "object"],
  "sphere": ["sphere", "ball"],
  "cube": ["cube", "block"],
  "large": ["large", "big"],
  "small": ["small", "tiny"],
  "metal": ["metallic", "metal", "shiny"],
  "rubber": ["rubber", "matte"],
}

type_attrs = {
    "Shape": [
      "cube", "sphere", "cylinder"
    ],
    "Color": [
      "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"
    ],
    "Relation": [
      "left", "right", "behind", "in_front_of"
    ],
    "Size": [
      "small", "large"
    ],
    "Material": [
      "rubber", "metal"
    ]
}
attr_types = {}
for k, v in type_attrs.items():
    for vv in v:
        attr_types[vv] = k

type_attrs_ext = {}
for k, v in type_attrs.items():
    v_new = [] + v
    for vv in v:
        syns = syn_attrs.get(vv, None)
        if syns is None:
            continue
        v_new.extend(syns)
    v_new = list(set(v_new))
    type_attrs_ext[k] = v_new
