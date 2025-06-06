import json
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript

class Questions:
    def __init__(self):
        self.questions = []
        # Load questions from JSON
        with open('questions/questions.json', 'r') as f:
            self.questions = json.load(f)

    def ask_question(self, index=0):
        q = self.questions[index]
        if q["type"] == "multiple_choice":
            self.display_multiple_choice_question(q)
        elif q["type"] == "multiple_choice_multiple":
            self.display_multi_choice(q)
        elif q["type"] == "matching":
            self.display_matching_question(q)
        elif q["type"] == "matching_dragdrop":
            self.display_matching_dragdrop(question_id=hash(q["question"]) % 10000, q=q)
        elif q["type"] == "open":
            self.display_open_question(q)
        else:
            print(f"⚠️ Unknown question type: {q.get('type')}")
        
    # === Display a Multiple-Choice Question ===
    def display_multiple_choice_question(self, q):
        print(q["question"])
        options = widgets.RadioButtons(options=q["options"])
        submit_button = widgets.Button(description="Antwoord indienen")
        output = widgets.Output()

        def on_submit(b):
            with output:
                clear_output()
                if options.value == q["answer"]:
                    print("✅ Correct!")
                else:
                    print(f"❌ Fout. Het juiste antwoord is: {q['answer']}")
        
        submit_button.on_click(on_submit)
        display(options, submit_button, output)

    # === Display an Open-Ended Question ===
    def display_open_question(self, q):
        print(q["question"])
        text_input = widgets.Text(placeholder='Geef je antwoord hier in...')
        submit_button = widgets.Button(description="Antwoord indienen")
        output = widgets.Output()

        def on_submit(b):
            with output:
                clear_output()
                user_answer = text_input.value.strip().lower()
                accepted_answers = [a.lower() for a in q["answer"]]
                if user_answer in accepted_answers:
                    print("✅ Correct!")
                else:
                    print(f"❌ Fout. De mogelijke antwoorden waren: {', '.join(q['answer'])}")

        submit_button.on_click(on_submit)
        display(text_input, submit_button, output)
        
    # === Display Multi-Answer Multiple Choice ===
    def display_multi_choice(self, q):
        print(q["question"])
        options = widgets.SelectMultiple(options=q["options"])
        button = widgets.Button(description="Submit")
        output = widgets.Output()

        def on_submit(b):
            with output:
                clear_output()
                selected = set(options.value)
                correct = set(q["answer"])
                if selected == correct:
                    print("✅ Correct!")
                else:
                    print(f"❌ Fout.\nJouw antwoord: {list(selected)}\nCorrecte antwoord: {list(correct)}")

        button.on_click(on_submit)
        display(options, button, output)
        
    def display_matching_question(self, q):
        print(q["question"])
        left_items = q["left"]
        right_items = q["right"]
        correct_pairs = q["answer"]

        dropdowns = {}
        for item in left_items:
            dropdowns[item] = widgets.Dropdown(
                options=[""] + right_items,
                description=item,
                layout=widgets.Layout(width='300px')
            )

        submit_button = widgets.Button(description="Submit")
        output = widgets.Output()

        def on_submit(b):
            with output:
                clear_output()
                user_answers = {k: dropdowns[k].value for k in dropdowns}
                if "" in user_answers.values():
                    print("⚠️ Please complete all matches.")
                    return

                if user_answers == correct_pairs:
                    print("✅ Correct!")
                else:
                    print("❌ Incorrect.")
                    print("Your answers:")
                    for k, v in user_answers.items():
                        print(f"  {k} → {v}")
                    print("\nCorrect answers:")
                    for k, v in correct_pairs.items():
                        print(f"  {k} → {v}")

        submit_button.on_click(on_submit)

        display(widgets.VBox(list(dropdowns.values()) + [submit_button, output]))
        
    def display_matching_dragdrop(self, question_id, q):
        left_items = q["left"]
        right_items = q["right"]
        correct_answer = q["answer"]

        html = f"""
        <style>
            .drag-container {{
                display: flex;
                gap: 50px;
                margin-bottom: 20px;
            }}
            .column {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .dropzone {{
                border: 2px dashed #aaa;
                padding: 10px;
                min-width: 100px;
                min-height: 20px;
            }}
            .draggable {{
                padding: 5px 10px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                cursor: move;
            }}
        </style>

        <div class="drag-container" id="dragmatch-{question_id}">
            <div class="column">
                <h4>Countries</h4>
                {''.join(f'<div>{country}<div class="dropzone" data-country="{country}"></div></div>' for country in left_items)}
            </div>
            <div class="column">
                <h4>Capitals</h4>
                {''.join(f'<div class="draggable" draggable="true" data-capital="{capital}">{capital}</div>' for capital in right_items)}
            </div>
        </div>

        <button onclick="checkMatch_{question_id}()">Submit</button>
        <div id="result-{question_id}"></div>

        <script>
            const draggables = document.querySelectorAll('#dragmatch-{question_id} .draggable');
            const dropzones = document.querySelectorAll('#dragmatch-{question_id} .dropzone');

            draggables.forEach(elem => {{
                elem.addEventListener('dragstart', e => {{
                    e.dataTransfer.setData('text/plain', e.target.dataset.capital);
                }});
            }});

            dropzones.forEach(zone => {{
                zone.addEventListener('dragover', e => {{
                    e.preventDefault();
                }});
                zone.addEventListener('drop', e => {{
                    e.preventDefault();
                    const capital = e.dataTransfer.getData('text/plain');
                    zone.textContent = capital;
                    zone.dataset.selected = capital;
                }});
            }});

            function checkMatch_{question_id}() {{
                const correct = {correct_answer};
                let correctCount = 0;
                let total = 0;
                dropzones.forEach(zone => {{
                    const country = zone.dataset.country;
                    const selected = zone.dataset.selected;
                    total++;
                    if (correct[country] === selected) {{
                        correctCount++;
                    }}
                }});
                const result = document.getElementById('result-{question_id}');
                if (correctCount === total) {{
                    result.innerHTML = "✅ Correct!";
                }} else {{
                    result.innerHTML = '❌' +  correctCount + ' out of ' + total + ' correct.';
                }}
            }}
        </script>
        """
        display(HTML(html))
        
    def stel_vraag(self, index=0):
        if 0 <= index < len(self.questions):
            self.ask_question(index)
        else:
            print("Index out of range. Please provide a valid question index.")
