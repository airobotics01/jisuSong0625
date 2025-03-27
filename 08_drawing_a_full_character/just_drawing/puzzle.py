import numpy as np
import os, json

script_path = os.path.abspath(__file__)
json_path = os.path.dirname(script_path)
json_path += "/asset/korean.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
    character_path = data["characters"]

def find_paths_by_name(character_name):
    for character in data["characters"]:
        if character["name"] == character_name:
            return character["path"]
    return None


def find_kind_by_name(character_name):
    for character in data["characters"]:
        if character["name"] == character_name:
            return character["kind"]
    return None

def get_coordinate(i):
    return [
        (i["start"][0], i["start"][1], i["start"][2]),
        (i["end"][0], i["end"][1], i["end"][2]),
    ]


def splitCharacter(a) -> list:
    a = int(ord(a))
    son = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', "ㅃ", 
           'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    mom = ['ㅏ','ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 
           'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 
           'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    support = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 
               'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 
               'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    arr = []
    if a >= ord('가'):
        a -= ord('가')
        arr.append(son[a // (len(mom) * (len(support) + 1))])
        a %= len(mom) * (len(support) + 1)
        arr.append(mom[a // (len(support) + 1)])
        a %= len(support) + 1
        if a:
            arr.append(support[a - 1])
    else:
        arr.append(chr(a))

    return arr


def makeStrokes(a) -> list:
    characters = splitCharacter(a)
    print(f"characters: {characters}")
    strokes = []
    
    if len(characters) == 1:
        stroke = find_paths_by_name(characters[0])
        for i in stroke:
            strokes.append(get_coordinate(i))
    
    elif len(characters) == 2:
        kind_info = find_kind_by_name(characters[1])
        print(f"kind: {kind_info[1]}")
        
        if kind_info is None:
            print(f"Warning: No kind info for '{characters[1]}'")
            return strokes
        
        elif kind_info[1] == 0:
            stroke = find_paths_by_name(characters[0])
            for i in stroke:
                strokes.append(get_coordinate(i))
            for i in strokes:
                i[0] = list(i[0])
                i[1] = list(i[1])
                i[0][1] = i[0][1] * 0.6
                i[1][1] = i[1][1] * 0.6
                i[0][2] = i[0][2] * 0.8 + 0.02
                i[1][2] = i[1][2] * 0.8 + 0.02
                i[0] = tuple(i[0])
                i[1] = tuple(i[1])
                
            # 중성 이동 적용 (예외 처리 추가)
            stroke = find_paths_by_name(characters[1])
            for i in stroke:
                strokes.append(get_coordinate(i))
        
        elif kind_info[1] == 1:
            stroke = find_paths_by_name(characters[0])
            for i in stroke:
                strokes.append(get_coordinate(i))
            for i in strokes:
                i[0] = list(i[0])
                i[1] = list(i[1])
                i[0][1] = i[0][1] * 0.8 + 0.02
                i[1][1] = i[1][1] * 0.8 + 0.02
                i[0][2] = i[0][2] * 0.6 + 0.06
                i[1][2] = i[1][2] * 0.6 + 0.06
                i[0] = tuple(i[0])
                i[1] = tuple(i[1])
            
            # 중성 이동 적용 (예외 처리 추가)
            stroke = find_paths_by_name(characters[1])
            for i in stroke:
                strokes.append(get_coordinate(i))
    
    elif len(characters) == 3:
        kind_info = find_kind_by_name(characters[1])
        print(f"kind: {kind_info[1]}")
        
        if kind_info is None:
            print(f"Warning: No kind info for '{characters[1]}'")
            return strokes
        
        elif kind_info[1] == 0:
            stroke = find_paths_by_name(characters[0])
            for i in stroke:
                strokes.append(get_coordinate(i))
            for i in strokes:
                i[0] = list(i[0])
                i[1] = list(i[1])
                i[0][1] = i[0][1] * 0.6
                i[1][1] = i[1][1] * 0.6
                i[0][2] = i[0][2] * 0.8 + 0.02
                i[1][2] = i[1][2] * 0.8 + 0.02
                i[0] = tuple(i[0])
                i[1] = tuple(i[1])
                
            # 중성 이동 적용 (예외 처리 추가)
            stroke = find_paths_by_name(characters[1])
            for i in stroke:
                strokes.append(get_coordinate(i))
        
        elif kind_info[1] == 1:
            stroke = find_paths_by_name(characters[0])
            for i in stroke:
                strokes.append(get_coordinate(i))
            for i in strokes:
                i[0] = list(i[0])
                i[1] = list(i[1])
                i[0][1] = i[0][1] * 0.8 + 0.02
                i[1][1] = i[1][1] * 0.8 + 0.02
                i[0][2] = i[0][2] * 0.6 + 0.06
                i[1][2] = i[1][2] * 0.6 + 0.06
                i[0] = tuple(i[0])
                i[1] = tuple(i[1])
            
            # 중성 이동 적용 (예외 처리 추가)
            stroke = find_paths_by_name(characters[1])
            for i in stroke:
                strokes.append(get_coordinate(i))
        
        for i in strokes:
            i[0] = list(i[0])
            i[1] = list(i[1])
            i[0][2] = i[0][2] * 0.65 + 0.07
            i[1][2] = i[1][2] * 0.65 + 0.07
            i[0] = tuple(i[0])
            i[1] = tuple(i[1])
        
        _strokes = []
        stroke = find_paths_by_name(characters[2])
        for i in stroke:
            _strokes.append(get_coordinate(i))
        for i in _strokes:
            i[0] = list(i[0])
            i[1] = list(i[1])
            i[0][1] = i[0][1] * 0.8 + 0.02
            i[1][1] = i[1][1] * 0.8 + 0.02
            i[0][2] = i[0][2] * 0.35
            i[1][2] = i[1][2] * 0.35
            i[0] = tuple(i[0])
            i[1] = tuple(i[1])
        strokes.extend(_strokes)

    dict = []
    for i in range(len(strokes)):
        dict.append({"start": strokes[i][0], "end": strokes[i][1]})
    return dict


