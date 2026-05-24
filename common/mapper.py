def langchain_messages_to_dict_format(messages):
    dicts = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            # already in dict format
            dicts.append(m)
        else:
            # convert from langchain Message object to dict
            role = "user" if m.type == "human" else m.type
            dicts.append({"role": role, "content": m.content})
    return dicts
