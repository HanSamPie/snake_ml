import gittools as git
import json

def version_control():
    commit = git.current_commit_hash(checkdirty=True, checktree=True)
    
    
    
    
    
    data = [
        {
            "name": "sathiyajith",
            "rollno": 56,
            "cgpa": 8.6,
            "phone": "9976770500"
        }
    ]

    json_str = json.dumps(data, indent=4)
    with open("sample.json", "w") as f:
        f.write(json_str)