

def generate_record_names():
    strings = []
    for subject in range(1, 12):
        for session in ["a", "b", "c", "d", "e"]:
            record_name = "T0{:0>2d}{}".format(subject, session)
            strings.append(record_name)
    return strings


print(generate_record_names())


