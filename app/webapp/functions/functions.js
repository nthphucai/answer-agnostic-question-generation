async function read_json_file(file_path) {
    let output = null;
    await fetch(file_path)
        .then(function (response) {
            output = response.json();
        }).catch(function (error) {
            console.log(error);
        })
    console.log(output);
    return output
}

function create_option(key, value) {
    let option = document.createElement("option");
    option.value = key;
    option.text = value;
    return option;
}

function show_history_content_choosen(content) {
    document.querySelector("#document").value = content;
}

function create_li_element(key, before) {
    let li_el = document.createElement("li");
    li_el.classList.add("dropdown-submenu");
    let a_el = document.createElement("a");
    a_el.classList.add("dropdown-item");
    if (before != "") {
        before += ">"
    }
    a_el.innerText = split_context(key);
    a_el.setAttribute("onclick", "show_history_content_choosen('" + before + key + "')")
    li_el.appendChild(a_el);
    return li_el;
}

function split_context(text) {
    text = text.split(" ");
    output = [""];
    for (let i = 0; i < text.length; i++) {
        if (output[output.length - 1].length < 30) {
            output[output.length - 1] += " " + text[i];
        } else {
            output[output.length - 1] = output[output.length - 1].trim();
            output.push(text[i]);
        }
    }
    return output.join("\n");
}

function show_history_choosing() {
    let history_ul = document.querySelector("#main-dropdown");
    history_ul.innerHTML = "";
    for (let [key, value] of Object.entries(history_content)) {
        let li_el = create_li_element(key, "");
        let ul_el = document.createElement("ul");
        ul_el.classList.add("dropdown-menu");
        li_el.appendChild(ul_el);
        history_ul.appendChild(li_el);
        for (let [key_1, value_1] of Object.entries(value)) {
            let li_el_1 = create_li_element(key_1, key);
            let ul_el_1 = document.createElement("ul");
            ul_el_1.classList.add("dropdown-menu");
            ul_el.appendChild(li_el_1);
            li_el_1.appendChild(ul_el_1);
            for (let [key_2, value_2] of Object.entries(value_1)) {
                let li_el_2 = create_li_element(key_2, key + ">" + key_1);
                let ul_el_2 = document.createElement("ul");
                ul_el_2.classList.add("dropdown-menu");
                ul_el_1.appendChild(li_el_2);
                li_el_2.appendChild(ul_el_2);
                for (let idx = 0; idx < value_2.length; idx++) {
                    let li_el_3 = create_li_element(value_2[idx], key + ">" + key_1 + ">" + key_2);
                    ul_el_2.appendChild(li_el_3);
                }
            }
        }
    }
}

function show_generate_type() {
    let domain = document.querySelector("#domain").value;
    let generate_type_el = document.querySelector("#generate-type");
    document.querySelector("#document").disabled = false;
    document.querySelector("#example").style.display = "none";
    document.querySelector("#example-label").style.display = "none";
    generate_type_el.innerHTML = "";
    for (let [key, value] of Object.entries(generate_domain[domain])) {
        if (key == "name") {
            continue;
        }
        let option = create_option(key, value);
        generate_type_el.add(option);
    }
    hide_fib_info();
    document.querySelector("#history-content").style.display = "none";
    if (domain == "none") {
        document.querySelector("#example").style.display = "none";
    } else if (domain == "history_textbook") {
        document.querySelector("#document").disabled = true;
        document.querySelector("#document").value = "";
        document.querySelector("#count_char").innerHTML = document.querySelector("#document").value.length;
        document.querySelector("#document").placeholder = "Chọn nội dung ở nút phía dưới!!!"
        document.querySelector("#history-content").style.display = "";
        show_history_choosing();
    } else {
        document.querySelector("#document").placeholder = "Nhập vào văn bản có độ dài từ 600 đến 1000 kí tự..."
    }
    show_generate_info();
}

function show_fib_info() {
    document.querySelector("#fib-num-blank").style.display = "";
    document.querySelector("#fib-label").style.display = "";
}

function hide_fib_info() {
    document.querySelector("#fib-num-blank").style.display = "none";
    document.querySelector("#fib-label").style.display = "none";
}

function create_example_options(idx) {
    let option = document.createElement("option");
    option.value = "example-" + idx.toString();
    option.text = "Ví dụ " + (idx + 1).toString();
    return option;
}

function show_generate_info() {
    document.querySelector("#example").style.display = "none";
    document.querySelector("#example-label").style.display = "none";
    let domain = document.querySelector("#domain").value;
    let generate_type = document.querySelector("#generate-type").value;
    if (generate_type == "fill-in-the-blank") {
        show_fib_info();
    } else {
        hide_fib_info();
        if ((domain != "history_textbook") && (domain != "none")) {
            let select_el = document.querySelector("#example");
            select_el.innerHTML = "";
            select_el.style.display = "";
            select_el.appendChild(create_option("none", ""));
            document.querySelector("#example-label").style.display = "";
            for (let idx = 0; idx < example[domain + "_" + generate_type].length; idx++) {
                select_el.appendChild(create_example_options(idx));
            }
        }
    }
}

function disable_everything() {
    document.querySelector('[id="document"]').disabled = true;
    document.querySelector('[id="generation"]').disabled = true;
    document.querySelector('[id="clear"]').disabled = true;
    document.querySelector('[id="choose_file"]').disabled = true;
}

function enable_everything() {
    document.querySelector('[id="document"]').disabled = false;
    document.querySelector('[id="generation"]').disabled = false;
    document.querySelector('[id="clear"]').disabled = false;
    document.querySelector('[id="choose_file"]').disabled = false;
}

function clear_everything() {
    log_generated_result(generated_result);
    document.querySelector('[id="document"]').value = null;
    document.querySelector('[id="choose_file"]').value = null;
    document.querySelector('[id="generation_error"]').style.display = "none";
    document.querySelector('[id="main_table"]').style.display = "none";
    document.querySelector('[id="clear"]').style.display = "none";
    document.querySelector('[id="domain_error"]').style.display = "none";
    document.querySelector('[id="runtime"]').style.display = "none";
    document.querySelector('[id="null_error"]').style.display = "none";
    document.querySelector('[id="input_error"]').style.display = "none";
    document.querySelector('[id="file_error"]').style.display = "none";
    document.querySelector('[id="waiting"]').style.display = "";
    document.querySelector('[id="stable_rocket"]').style.display = "";
    document.querySelector('[id="flying_rocket"]').style.display = "none";
}

function show_example_text() {
    let value = document.querySelector("#example").value;
    if (value != "none") {
        value = value.split("-")[1];
        let domain = document.querySelector("#domain").value;
        let generate_type = document.querySelector("#generate-type").value;
        document.querySelector("#document").value = example[domain + "_" + generate_type][Number.parseInt(value)];
    } else {
        document.querySelector("#document").value = "";
    }
    document.querySelector("#count_char").innerHTML = document.querySelector("#document").value.length;
}

function read_content() {
    log_generated_result();
    var fileToLoad = document.getElementById("choose_file").files[0];
    document.querySelector('[id="document"]').value = null;

    if (fileToLoad.size > maximumSize) {
        document.querySelector('[id="file_error"]').style.display = "";
    } else {
        var fileReader = new FileReader();
        fileReader.onload = function (fileLoadedEvent) {
            var textFromFileLoaded = fileLoadedEvent.target.result;
            document.getElementById("document").value = textFromFileLoaded;
        };
        fileReader.readAsText(fileToLoad, "UTF-8");
    }
    document.querySelector("#count_char").innerHTML = document.querySelector("#document").value.length;
}

function toUpperString(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function create_tr_output(idx, prefix_id, text_content, text_style) {
    let td_el = document.createElement("td");
    td_el.style.textAlign = text_style;
    let div_el = document.createElement("div");
    div_el.innerText = text_content;
    td_el.appendChild(div_el);
    return td_el;
}

function create_checkbox_el(idx) {
    let td_el = document.createElement("td");
    td_el.style.textAlign = "center";
    let div_el = document.createElement("div");
    div_el.classList.add("form-check");
    let checkbox_btn = document.createElement("input");
    checkbox_btn.type = "checkbox";
    checkbox_btn.classList.add("form-check-input");
    checkbox_btn.value = "";
    checkbox_btn.id = "check_" + idx.toString();
    checkbox_btn.setAttribute("onclick", "changeChecked(" + idx + ")");
    div_el.appendChild(checkbox_btn);
    td_el.appendChild(div_el);
    return td_el;
}

function create_comment_el(idx) {
    let td_el = document.createElement("td");
    let textarea_el = document.createElement("textarea");
    textarea_el.classList.add("form-control");
    textarea_el.id = "comment_" + idx.toString();
    td_el.appendChild(textarea_el);
    return td_el
}

function show_simple_question(output) {
    tbody_tb = document.querySelector('[id="result_tbody"]');
    idx = -1
    for (let j = 0; j < output.length; j++) {
        context = output[j]["context"];
        sub_output = output[j]["results"];
        for (let i = 0; i < sub_output.length; i++) {
            idx += 1
            item = sub_output[i]
            let tr_el = document.createElement("tr");
            tr_el.id = "for_id_" + idx.toString();
            tr_el.appendChild(create_tr_output(idx, "id_", (idx + 1).toString(), "center"));
            tr_el.appendChild(create_tr_output(idx, "question_", toUpperString(item["question"]), "left"));
            tr_el.appendChild(create_tr_output(idx, "answer_", toUpperString(item["answer"]), "left"));
            tr_el.appendChild(create_checkbox_el(idx));
            tr_el.appendChild(create_comment_el(idx));

            tbody_tb.appendChild(tr_el);

            generated_result["results"].push({
                "context": context,
                "question": item["question"],
                "answer": item["answer"],
                "check": 0,
                "comment": "",
            })
        }
    }
}

function show_multiple_choice(output) {
    tbody_tb = document.querySelector('[id="result_tbody"]');
    idx = -1
    for (let j = 0; j < output.length; j++) {
        context = output[j]["context"];
        sub_output = output[j]["results"];
        for (let i = 0; i < sub_output.length; i++) {
            idx += 1;
            item = sub_output[i];

            let tr_el = document.createElement("tr");
            tr_el.id = "for_id_" + idx.toString();
            tr_el.appendChild(create_tr_output(idx, "id_", (idx + 1).toString(), "center"));
            tr_el.appendChild(create_tr_output(idx, "question_", toUpperString(item["question"]), "left"));

            let td_el = document.createElement("td");
            td_el.id = "answer_" + idx.toString();
            td_el.style.textAlign = "left";
            let div_el = document.createElement("div");
            let p_el = null;
            for (let idc = 0; idc < item["options"].length; idc++) {
                p_el = document.createElement("p");
                p_el.style.marginBottom = 0;
                p_el.innerText = toUpperString(item["options"][idc]);
                div_el.appendChild(p_el);
                p_el = null;
            }
            p_el = document.createElement("p");
            p_el.style.marginTop = "10px";
            p_el.innerHTML = "Đáp án đúng: <b>" + item["answer"] + "</b>";
            div_el.appendChild(p_el);
            td_el.appendChild(div_el);
            tr_el.appendChild(td_el);
            div_el = null;
            td_el = null;

            tr_el.appendChild(create_checkbox_el(idx));
            tr_el.appendChild(create_comment_el(idx));

            tbody_tb.appendChild(tr_el);

            generated_result["results"].push({
                "context": context,
                "question": item["question"],
                "answer": item["answer"],
                "check": 0,
                "comment": "",
            })
        }
    }
}

async function quest_gen() {
    log_generated_result();
    disable_everything();
    document.querySelector('[id="runtime"]').style.display = "none";
    document.querySelector('[id="result_tbody"]').innerHTML = "";
    document.querySelector('[id="generation_error"]').style.display = "none";
    document.querySelector('[id="null_error"]').style.display = "none";
    document.querySelector('[id="main_table"]').style.display = "none";
    document.querySelector('[id="clear"]').style.display = "none";
    document.querySelector("#domain_error").style.display = "none";
    document.querySelector('[id="input_error"]').style.display = "none";
    document.querySelector('[id="file_error"]').style.display = "none";
    document.querySelector('[id="waiting"]').style.display = "";
    document.querySelector('[id="flying_rocket"]').style.display = "";
    document.querySelector('[id="stable_rocket"]').style.display = "none";
    let text = document.querySelector('[id="document"]').value;
    let domain = document.querySelector("#domain").value;
    let generate_type = document.querySelector("#generate-type").value;
    try {
        if (domain == "none") {
            document.querySelector("#domain_error").style.display = "";
            document.querySelector('[id="stable_rocket"]').style.display = "";
            document.querySelector('[id="flying_rocket"]').style.display = "none";
        } else if ((text == "") && (generate_type != "fill-in-the-blank")) {
            document.querySelector('[id="input_error"]').style.display = "";
            document.querySelector('[id="stable_rocket"]').style.display = "";
            document.querySelector('[id="flying_rocket"]').style.display = "none";
        } else {
            let input = null;
            let output = null;
            let time = null;
            if (domain == "history_textbook") {
                input = JSON.stringify({"task": generate_type, "section": text});
                await axios({
                    responseType: "json",
                    method: "post",
                    url: api_url["generate_from_book"],
                    headers: {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                    data: input,
                    timeout: timeout,
                }).then(function (response) {
                    output = response;
                }).catch(function (error) {
                    console.log(error);
                });
                time = output.data["time"];
                output = output.data["data"];
            } else if (generate_type != "fill-in-the-blank") {
                input = JSON.stringify({"task": generate_type, "domain": domain, "context": text});
                await axios({
                    responseType: "json",
                    method: "post",
                    url: api_url["generate_api"],
                    headers: {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                    data: input,
                    timeout: timeout,
                }).then(function (response) {
                    output = response;
                }).catch(function (error) {
                    console.log(error);
                });
                time = output.data["time"];
                output = output.data["data"];
            } else {
                let num_blank = document.querySelector("#fib-num-blank").value;
                input = JSON.stringify({"context": text, "num_blank": num_blank})
                await axios({
                    responseType: "json",
                    method: "post",
                    url: api_url["fib_api"],
                    headers: {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                    data: input,
                    timeout: timeout,
                }).then(function (response) {
                    output = response;
                }).catch(function (error) {
                    console.log(error);
                });
                time = output.data["time"];
                output = [output.data];
            }
            try {
                if (output == null) {
                    document.querySelector('[id="generation_error"]').style.display = "";
                    document.querySelector('[id="stable_rocket"]').style.display = "";
                    document.querySelector('[id="flying_rocket"]').style.display = "none";
                } else {
                    document.querySelector('[id="waiting"]').style.display = "none";
                    document.querySelector('[id="main_table"]').style.display = "";
                    document.querySelector('[id="clear"]').style.display = "";
                    generated_result["task"] = generate_type;
                    generated_result["domain"] = domain;
                    generated_result["results"] = [];
                    if (output.length < 1) {
                        document.querySelector('[id="null_error"]').style.display = "";
                        document.querySelector('[id="waiting"]').style.display = "";
                        document.querySelector('[id="stable_rocket"]').style.display = "";
                        document.querySelector('[id="flying_rocket"]').style.display = "none";
                        document.querySelector('[id="main_table"]').style.display = "none";
                        document.querySelector('[id="clear"]').style.display = "none";
                    } else {
                        if (generate_type != "simple-question") {
                            show_multiple_choice(output);
                        } else {
                            show_simple_question(output);
                        }
                        generated_result["time"] = time;
                        document.querySelector('[id="runtime"]').innerText = "Runtime: " + Number((time).toFixed(2)).toString() + "ms";
                        document.querySelector('[id="runtime"]').style.display = "";
                    }
                }
            } catch {
                document.querySelector('[id="main_table"]').style.display = "none";
                document.querySelector('[id="clear"]').style.display = "none";
                document.querySelector('[id="generation_error"]').style.display = "";
                document.querySelector('[id="waiting"]').style.display = "";
                document.querySelector('[id="stable_rocket"]').style.display = "";
                document.querySelector('[id="flying_rocket"]').style.display = "none";
            }
        }
    } catch {
        document.querySelector('[id="main_table"]').style.display = "none";
        document.querySelector('[id="clear"]').style.display = "none";
        document.querySelector('[id="generation_error"]').style.display = "";
        document.querySelector('[id="waiting"]').style.display = "";
        document.querySelector('[id="stable_rocket"]').style.display = "";
        document.querySelector('[id="flying_rocket"]').style.display = "none";
    }
    enable_everything();
}

function changeChecked(number) {
    number = Number.parseInt(number);
    if (generated_result["results"][number]["check"] == 1) {
        generated_result["results"][number]["check"] = 0;
    } else {
        generated_result["results"][number]["check"] = 1;
    }
}

function prepare_comment_log() {
    for (let i = 0; i < generated_result["results"].length; i++) {
        text = document.querySelector('[id="comment_' + i.toString() + '"]').value;
        generated_result["results"][i]["comment"] = text;
    }
}

async function log_generated_result() {
    if (Object.keys(generated_result) != 0) {
        prepare_comment_log();
        let output = null;
        await axios({
            responseType: "json",
            method: "post",
            url: api_url["feedback_api"],
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            data: JSON.stringify(generated_result),
            timeout: timeout,
        }).then(function (response) {
            output = response;
        }).catch(function (error) {
            console.log(error);
        });
        generated_result = {};
    }
}