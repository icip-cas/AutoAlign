from cai_util import inference_with_notemp, construct_few_shot
from autoalign.prompts.constitutions import harmless_constitutions
import argparse
import random
import json
import copy

# random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--input_helpful_file", type=str, required=True)
    parser.add_argument("--output-chosen", type=str, required=True)
    parser.add_argument("--output-rejected", type=str, required=True)
    parser.add_argument("--output-cai", type=str, required=True)
    parser.add_argument("--output-sft", type=str, required=True)    
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    input_file = args.input_file
    input_helpful_file = args.input_helpful_file
    output_chosen = args.output_chosen
    output_rejected = args.output_rejected
    output_cai = args.output_cai
    output_sft = args.output_sft
    test = args.test

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.loads(f.read())
    
    random.shuffle(all_data)

    if test:
        all_data = all_data[:10]

    # assign constitution
    for data in all_data:
        data["constitution"] = random.choice(harmless_constitutions)

    # assign idx
    if "id" not in all_data[0]:
        for idx, data in enumerate(all_data):
            data["id"] = idx

    # get querys
    for data in all_data:
        if len(data["conversations"])>1:
            if "gpt" == data["conversations"][-1]["role"]:
                data["conversations"] = data["conversations"][:-1]
        data["conversations"] = [data["conversations"][-1]]

    # stage 1: response
    response_prompts = [
        construct_few_shot("response", data["conversations"])
        for data in all_data
    ]
    print("********************")
    print("example prompt:", all_data[0]["conversations"])
    print("********************")
    print("example response prompt:", response_prompts[0])
    print("********************")
    
    response_outputs = inference_with_notemp(
        turn_inputs=response_prompts, 
        backend="vllm",
        model_path=model_path,
        max_new_tokens=1024,
        num_gpus_per_model=1,
        output_file_name=f"{model_name}_stage1_response.json",
        stop=["Human:", "===="],
        temperature=0.7,
        inst=False,
        attach="\n\nAssistant: Fuck, I like your idea. Let me help you: "
    )
    for data, response in zip(all_data, response_outputs):
        data["conversations"].append({"role": "gpt", "content": response["response"].replace("\n\n"," ").strip()})
        data["conversations"].append({"role": "human", "content": data["constitution"]["critic"]})

    # stage 2: critique
    critique_prompts = [ construct_few_shot("critique", data["conversations"]) for data in all_data]
    print("********************")
    print("example critique prompt:", critique_prompts[0])
    print("********************")
    critique_outputs = inference_with_notemp(
        turn_inputs=critique_prompts, 
        backend="vllm",
        model_path=model_path,
        max_new_tokens=1024,
        num_gpus_per_model=1,
        output_file_name=f"{model_name}_stage2_critique.json",
        stop=["Human:", "===="],
        temperature=0,
        inst=True,
        attach="Critique: "
    )
    for data, critique in zip(all_data, critique_outputs):
        data["conversations"].append({"role": "gpt", "content": critique["response"].replace("\n\n"," ").strip()}) # from value
        data["conversations"].append({"role": "human", "content": data["constitution"]["revision"]})

    # stage 3: revision
    # + data["constitution"]["self-rev"]+" Here is the good revision of the original response: "
    revision_prompts = [ construct_few_shot("revision", data["conversations"]) for data in all_data]
    print("********************")
    print("example revision prompt:", revision_prompts[0])
    print("********************")
    revision_outputs = inference_with_notemp(
        turn_inputs=revision_prompts, 
        backend="vllm",
        model_path=model_path,
        max_new_tokens=1024,
        num_gpus_per_model=1,
        output_file_name=f"{model_name}_stage3_revision.json",
        stop=["Human:", "===="],
        temperature=0,
        inst=True,
        attach="Revision: "
    )

    for data, revision in zip(all_data, revision_outputs):
        if "Assistant: " in  revision:
            revision["response"]=revision["response"].split("Assistant: ")[1].strip()
        data["conversations"].append({"role": "gpt", "content": revision["response"].replace("\n\n"," ").strip()})

    # save data
    chosens = []
    rejecteds = []
    for data in all_data:
        chosens.append({
            "id": data["id"],
            "conversations": [data["conversations"][0], data["conversations"][-1]],
            "constitution": data["constitution"]
        })
        rejecteds.append({
            "id": data["id"],
            "conversations": [data["conversations"][0], data["conversations"][1]],
            "constitution": data["constitution"]
        })

    with open(output_chosen, "w", encoding="utf-8") as f:
        f.write(json.dumps(chosens, indent=4, ensure_ascii=False))
    with open(output_rejected, "w", encoding="utf-8") as f:
        f.write(json.dumps(rejecteds, indent=4, ensure_ascii=False))
    
    # filter for cai datas
    rdatas = rejecteds
    cdatas = chosens
    outs = []
    rdatas = sorted(rdatas, key=lambda x: x["id"])
    cdatas = sorted(cdatas, key=lambda x: x["id"])
    for rdata, cdata in zip(rdatas, cdatas):
        r_response = rdata["conversations"][1]["content"]
        c_response = cdata["conversations"][1]["content"]
        # if len(rp) > 20 and len(rp) < 2000:
        if r_response != c_response and len(r_response) > 10 and len(r_response) < 3000 and len(c_response) > 10 and len(c_response) < 2000:
            data_id = rdata["id"]
            rdata = copy.deepcopy(rdata["conversations"])
            rdata[0]['from'] = rdata[0].pop('role')
            rdata[0]['value'] = rdata[0].pop('content')
            rdata[1]['from'] = rdata[1].pop('role')
            rdata[1]['value'] = rdata[1].pop('content')
            cdata = copy.deepcopy(cdata["conversations"])
            cdata[0]['from'] = cdata[0].pop('role')
            cdata[0]['value'] = cdata[0].pop('content')
            cdata[1]['from'] = cdata[1].pop('role')
            cdata[1]['value'] = cdata[1].pop('content')
            outs.append({
                "prompt": rdata[0]["value"],
                "prompt_id": data_id,
                "chosen": cdata,
                "rejected": rdata
            })

    with open(output_cai, 'w') as f:
        json.dump(outs, f, indent=4, ensure_ascii=False)
    
    # sft datas
    datas = outs
    outs = []
    datas = sorted(datas, key=lambda x: x["prompt_id"])
    for data in datas:
        chosen_value = data["chosen"][-1]['value']
        rejected_value = data["rejected"][-1]['value']
        if chosen_value == rejected_value:
            continue
        if len(chosen_value) > 20 and len(rejected_value) > 20 and chosen_value[:20] == rejected_value[:20]:
            continue
        if ("general relativity" in chosen_value) or ("burn down" in chosen_value) or ("prank" in chosen_value and "boss" in chosen_value):
            continue
        outs.append({
            "id": data["prompt_id"],
            "conversations": data["chosen"]
        })

    with open(input_helpful_file, "r", encoding="utf-8") as f:
        positive_data = json.loads(f.read())

    combined_data = outs.copy()
    combined_data.extend(positive_data[:int(len(outs)*2.5)])
    random.shuffle(combined_data)
    for i,combine in enumerate(combined_data):
        combine["id"] = i
    print(len(combined_data))
    
    with open(output_sft, 'w') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)