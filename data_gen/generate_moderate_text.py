import os
import os.path as osp
import json
import numpy as np
import tqdm
import string
# import inflect
# p_engine = inflect.engine()

no_color = False
add_object_number = False

combine_list = [None, "direction", "label", "color"]
reserve_list = [True, False]
target_type_list = ["non-human", "human"]
subject_type_list = ["subject", "object"]
target_list = ["pose", "target", "position", "location", "place", "spot", "destination"]
subject_list = ["I", "we", "you", "they", "he", "she"]
object_list = ["me", "us", "you", "them", "him", "her"]
posessive_list = ["my", "our", "your", "their", "his", "him"]
verb_list = ["be", "located", "lied", "positioned", "situated"]

def pluralize(word, times=1):
    if times == 1:
        return word
    else:
        return p_engine.plural(word)

def opposite_direction(d):
    if d == "north":
        return "south"
    elif d == "south":
        return "north"
    elif d == "east":
        return "west"
    elif d == "west":
        return "east"
    return "below"

def add_of(d):
    if d == "below":
        return d
    else:
        return d + " of"

def be_verb(subject):
    if subject == "I":
        return "am"
    if subject in ["you", "they", "we"]:
        return "are"
    else:
        return "is"
    
def num_to_text(value):
    if value == 1:
        return "the"
    elif value == 2:
        return "two"
    elif value == 3:
        return "three"
    elif value == 4:
        return "four"
    elif value == 5:
        return "five"
    else:
        return "six"
    
def function_one(number):
    if number == 1:
        return "one"
    else:
        return "ones"
    

def generate_moderate_text(pose):
    descriptions = pose.descriptions
    combine_choice_list = [None, "direction"]
    pose_dict = {}
    desc_ele = []
    desc_dict = {}

    direction_list = []
    color_list = []
    label_list = []

    direction_index = []
    color_index = []
    label_index = []
    for sen_idx, descr in enumerate(descriptions):
        desc_ele.append([descr.direction, descr.object_color_text, descr.object_label])
        if descr.direction not in direction_list:
            direction_list.append(descr.direction)
            direction_index.append([sen_idx])
        else:
            for dir_ind in range(len(direction_index)):
                if descr.direction == direction_list[dir_ind]:
                    direction_index[dir_ind].append(sen_idx)

        if descr.object_color_text in color_list:
            if "color" not in combine_choice_list:
                combine_choice_list.append("color")
            for col_ind in range(len(color_index)):
                if descr.object_color_text == color_list[col_ind]:
                    color_index[col_ind].append(sen_idx)
        else:
            color_list.append(descr.object_color_text)
            color_index.append([sen_idx])

        if descr.object_label in label_list:
            if "label" not in combine_choice_list:
                combine_choice_list.append("label")
            for lab_ind in range(len(label_index)):
                if descr.object_label == label_list[lab_ind]:
                    label_index[lab_ind].append(sen_idx)
        else:
            label_list.append(descr.object_label)
            label_index.append([sen_idx])
    
    # all_direction_index.append(direction_index)
    # all_color_index.append(color_index)
    # all_label_index.append(label_index)

    pose_dict["combine"] = combine_choice_list[np.random.choice(range(len(combine_choice_list)), size = 1, replace = False)[0]]
    if pose_dict["combine"] is None:
        for idx, sen in enumerate(desc_ele):
            desc_dict[idx] = sen
    elif pose_dict["combine"] == "direction":
        for direction in direction_list:
            desc_dict[direction] = []
        for sen in desc_ele:
            desc_dict[sen[0]].append(sen)
    elif pose_dict["combine"] == "color":
        for color in color_list:
            desc_dict[color] = []
        for sen in desc_ele:
            desc_dict[sen[1]].append(sen)
    elif pose_dict["combine"] == "label":
        for label in label_list:
            desc_dict[label] = []
        for sen in desc_ele:
            desc_dict[sen[2]].append(sen)
    else:
        raise "ERROR"

    pose_dict["reserve"] = reserve_list[np.random.choice(2, size = 1, replace = False)[0]]
    human_index = np.random.choice(len(subject_list), size = 1, replace = False)[0]
    pose_dict["subject"] = subject_list[human_index]
    pose_dict["object"] = object_list[human_index]

    pose_dict["target_type"] = []
    pose_dict["target"] = []

    for _ in desc_dict.values():
        target_type = target_type_list[np.random.choice(2, size = 1, replace = False)[0]]
        if target_type == "non-human":
            pose_dict["target"].append(target_list[np.random.choice(range(len(target_list)), size = 1, replace = False)[0]])
        else:
            target_type = subject_type_list[np.random.choice(2, size = 1, replace = False)[0]]
            if target_type == "subject":
                pose_dict["target"].append(pose_dict["subject"])
            else:
                pose_dict["target"].append(pose_dict["object"])

        pose_dict["target_type"].append(target_type)


    pose_dict["data"] = desc_dict
    pose_dict["raw_data"] = desc_ele

    text = []
    reserve = pose_dict["reserve"]
    subject = pose_dict["subject"]
    object0 = pose_dict["object"]
    target = pose_dict["target"]
    target_type = pose_dict["target_type"]
    data_key = [key for key in pose_dict["data"].keys()]
    data_value = [value for value in pose_dict["data"].values()]
    if pose_dict["combine"] is None:
        potential_index = [[i] for i in range(len(target))]
        for index in range(len(target)):
            target_type_i = target_type[index]
            d_i, c_i, l_i = data_value[index]
            if target_type_i == "non-human":
                if not reserve:
                    d_i = d_i
                    s_i = "The " + target[index]
                    o_i = " ".join(["the", c_i, l_i])                                 
                else:
                    d_i = opposite_direction(d_i)
                    s_i = " ".join(["The", c_i, l_i])
                    o_i = "the " + target[index]
                # v_i_choice = np.random.choice(verb_list, size = 1, replace = False)[0]
                v_i = "is"
                # if v_i_choice != "be":
                #     v_i += " " + v_i_choice
                v_i += " " + add_of(d_i)
            elif target_type_i == "subject":
                s_i = subject.title()
                v_i = " ".join([be_verb(subject), add_of(d_i)])
                o_i = " ".join(["the", c_i, l_i])
            else:
                s_i = " ".join(["The", c_i, l_i])
                v_i = " ".join(["is", add_of(opposite_direction(d_i))])
                o_i = object0
            text.append(" ".join([s_i, v_i, o_i])+".")
    
    elif pose_dict["combine"] == "direction":
        potential_index = direction_index
        for index in range(len(target)):
            target_type_i = target_type[index]
            # d_s, c_s, t_s = zip(*data_value[index])
            # d_i = data_key[index]
            # p_i_list = []
            label_dict = {}
            for d_i, c_i, l_i in data_value[index]:
                # p_i_list.append(" ".join(["the", c_i, l_i]))
                if l_i not in label_dict.keys():
                    label_dict[l_i] = {c_i: 1}
                elif c_i not in label_dict[l_i].keys():
                    label_dict[l_i][c_i] = 1
                else:
                    label_dict[l_i][c_i] += 1
                
            # if len(p_i_list) == 1:
            #     p_i = " ".join(["the",  data_value[index][0][1],  data_value[index][0][2]])
            # elif len(p_i_list) == 2:
            #     if len(label_dict) == 1:
            #         if len(label_dict[l_i]) == 1:
            #             p_i = "two " + " ".join([data_value[index][0][1],  data_value[index][0][2]])
            #         else:
            #             p_i = " ".join(["the", data_value[index][0][1]]) + " and " + \
            #                   " ".join(["the",  data_value[index][1][1],  data_value[index][1][2]])
            # else:
            label_text_list = []
            for label, value in label_dict.items():
                color_text_list = []
                label_times = 0 
                for color, times in value.items():
                    color_text_list.append(num_to_text(times) + " " + color)
                    label_times += times
                if len(color_text_list) == 1:
                    color_text = color_text_list[0]
                elif len(color_text_list) == 2:
                    color_text = " and ".join(color_text_list)
                else:
                    color_text = ", ".join(color_text_list[:-1])
                    color_text = " and ".join([color_text, color_text_list[-1]])

                label_text_list.append(color_text + " " + pluralize(label, label_times))

            p_i_subject = "they"
            if len(label_dict) == 1:
                p_i = label_text_list[0]
                if label_times == 1:
                    p_i_subject = "he"
            elif len(label_dict) == 2:
                p_i = label_text_list[0] + " and " + label_text_list[1]
            else:
                p_i = ", ".join(label_text_list[:-1])
                p_i += " and " + label_text_list[-1]
                    

            if target_type_i == "non-human":
                if not reserve:
                    s_i = "The " + target[index]
                    o_i = p_i
                    v_i = "is"
                else:
                    d_i = opposite_direction(d_i)
                    s_i = p_i 
                    o_i = "the " + target[index]
                    v_i = be_verb(p_i_subject) 
                # v_i_choice = np.random.choice(verb_list, size = 1, replace = False)[0]
                # if v_i_choice != "be":
                #     v_i += " " + v_i_choice
                v_i += " " + add_of(d_i)
            elif target_type_i == "subject":
                s_i = subject
                v_i = " ".join([be_verb(subject), add_of(d_i)])
                o_i = p_i
            else:
                s_i = p_i 
                v_i = " ".join([be_verb(p_i_subject), add_of(opposite_direction(d_i))])
                o_i = object0
            s_i = s_i[0].title() + s_i[1:]
            text.append(" ".join([s_i, v_i, o_i]) + ".")

    elif pose_dict["combine"] == "color":
        potential_index = color_index
        for index in range(len(target)):
            target_type_i = target_type[index]
            label_dict = {}
            direction_dict = {}
            for d_i, c_i, l_i in data_value[index]:
                # p_i_list.append(" ".join(["the", c_i, l_i]))
                if l_i not in label_dict.keys():
                    label_dict[l_i] = {d_i : 1}
                elif d_i not in label_dict[l_i].keys():
                    label_dict[l_i][d_i] = 1
                else:
                    label_dict[l_i][d_i] += 1

                if d_i not in direction_dict.keys():
                    direction_dict[d_i] = {l_i : 1}
                elif l_i not in direction_dict[d_i].keys():
                    direction_dict[d_i][l_i] = 1
                else:
                    direction_dict[d_i][l_i] += 1
                
            label_text_list = []
            for label, value in label_dict.items():
                number = sum([num for num in value.values()])
                label_text_list.append(num_to_text(number) + " " + pluralize(label, number))
            if len(label_dict) == 1:
                p_i = label_text_list[0]
                if number == 1:
                    pre = ""
                else:
                    p_i = label_text_list[0]
                    pre = " ".join([p_i, "are", c_i]) + "."
                    pre = pre[0].title() + pre[1:] 
            else:
                if len(label_dict) == 2:
                    p_i = label_text_list[0] + " and " + label_text_list[1]
                else:
                    p_i = ", ".join(label_text_list[:-1])
                    p_i += " and " + label_text_list[-1]
                pre = " ".join([p_i, "are", c_i]) + "."
                pre = pre[0].title() + pre[1:]

            direction_text_list = []
            for d_i, value in direction_dict.items():
                label_text_list = [] 
                for label, times in value.items():
                    if pre == "":
                        label_text_list.append(" ".join([num_to_text(times), c_i, pluralize(label, times)])) 
                    elif len(label_dict[label]) != 1 and times == 1:
                        label_text_list.append("one " + label)
                    else:
                        label_text_list.append(num_to_text(times) + " " + pluralize(label, times))

                p_i_subject = "they"
                if len(label_text_list) == 1:
                    p_i = label_text_list[0]
                    if times == 1:
                        p_i_subject = "he"
                elif len(label_text_list) == 2:
                    p_i = label_text_list[0] + " and " + label_text_list[1]
                else:
                    p_i = ", ".join(label_text_list[:-1])
                    p_i += " and " + label_text_list[-1]
                        
                if target_type_i == "non-human":
                    if not reserve:
                        s_i = "The " + target[index]
                        o_i = p_i
                        v_i = "is"
                    else:
                        d_i = opposite_direction(d_i)
                        s_i = p_i 
                        o_i = "the " + target[index]
                        v_i = be_verb(p_i_subject) 
                    # v_i_choice = np.random.choice(verb_list, size = 1, replace = False)[0]
                    # if v_i_choice != "be":
                    #     v_i += " " + v_i_choice
                    v_i += " " + add_of(d_i)
                elif target_type_i == "subject":
                    s_i = subject
                    v_i = " ".join([be_verb(subject), add_of(d_i)])
                    o_i = p_i
                else:
                    s_i = p_i 
                    v_i = " ".join([be_verb(p_i_subject), add_of(opposite_direction(d_i))])
                    o_i = object0

                s_i = s_i[0].title() + s_i[1:]
                direction_text_list.append(" ".join([s_i, v_i, o_i]) + ".")
            
            if pre != "":
                text.append(" ".join([pre] + direction_text_list))
            else:
                text.append(" ".join(direction_text_list))
    else:
        potential_index = label_index
        for index in range(len(target)):
            target_type_i = target_type[index]
            color_dict = {}
            count = 0
            for d_i, c_i, l_i in data_value[index]:
                count += 1
                # p_i_list.append(" ".join(["the", c_i, l_i]))
                if c_i not in color_dict.keys():
                    color_dict[c_i] = {d_i : 1}
                elif d_i not in color_dict[c_i].keys():
                    color_dict[c_i][d_i] = 1
                else:
                    color_dict[c_i][d_i] += 1
            
            if count == 1: # same as None situation
                pre = ""
                if target_type_i == "non-human":
                    if not reserve:
                        s_i = "The " + target[index]
                        o_i = " ".join(["the", c_i, l_i])                                 
                    else:
                        d_i = opposite_direction(d_i)
                        s_i = " ".join(["The", c_i, l_i])
                        o_i = "the " + target[index]
                    # v_i_choice = np.random.choice(verb_list, size = 1, replace = False)[0]
                    v_i = "is"
                    # if v_i_choice != "be":
                    #     v_i += " " + v_i_choice
                    v_i += " " + add_of(d_i)
                elif target_type_i == "subject":
                    s_i = subject.title()
                    v_i = " ".join([be_verb(subject), add_of(d_i)])
                    o_i = " ".join(["the", c_i, l_i])
                else:
                    s_i = " ".join(["The", c_i, l_i])
                    v_i = " ".join(["is", add_of(opposite_direction(d_i))])
                    o_i = object0
            else:
                pre =  " ".join(["There are", num_to_text(count), pluralize(l_i, num_to_text(count)) + "."])
                direction_text_list = []
                for c_i, value in color_dict.items():
                    for d_i, times in value.items():
                        p_i = " ".join([num_to_text(times), c_i, function_one(times)]) 

                        p_i_subject = "they" if times > 1 else "he"

                        if target_type_i == "non-human":
                            if not reserve:
                                s_i = "The " + target[index]
                                o_i = p_i
                                v_i = "is"
                            else:
                                d_i = opposite_direction(d_i)
                                s_i = p_i 
                                o_i = "the " + target[index]
                                v_i = be_verb(p_i_subject) 
                            # v_i_choice = np.random.choice(verb_list, size = 1, replace = False)[0]
                            # if v_i_choice != "be":
                            #     v_i += " " + v_i_choice
                            v_i += " " + add_of(d_i)
                        elif target_type_i == "subject":
                            s_i = subject
                            v_i = " ".join([be_verb(subject), add_of(d_i)])
                            o_i = p_i
                        else:
                            s_i = p_i 
                            v_i = " ".join([be_verb(p_i_subject), add_of(opposite_direction(d_i))])
                            o_i = object0

                        s_i = s_i[0].title() + s_i[1:]
                        direction_text_list.append(" ".join([s_i, v_i, o_i]) + ".")

                

            if pre != "":
                text.append(" ".join([pre] + direction_text_list))
            else:
                text.append(" ".join([s_i, v_i, o_i]) + ".")
    
    size = len(text)
    permute_index = np.random.choice(list(range(size)), size = size, replace = False)
    final_index = [potential_index[i] for i in permute_index]
    final_text = [text[i] for i in permute_index]

    return final_text, final_index