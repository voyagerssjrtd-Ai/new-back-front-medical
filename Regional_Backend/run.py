from workflow import app
import utils.security as security
import json

def calling_langgarph(user_input: str):
    
    result = app.invoke({
        "query": user_input
    })

    if(result['intent'] == "chat"):
        raw_output = result['output']
        cleaned = raw_output.strip("`").replace("json", "", 1).strip()
        data = json.loads(cleaned)
        output_json = json.dumps(data, indent=4)
        return output_json
    else:
        output_json = json.dumps(result, indent=4)
        return output_json
    
user_input = input("Enter request: ")
print(calling_langgarph(user_input))