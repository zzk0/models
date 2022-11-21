from openprompt.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        text_a = "Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as Teachers. My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is Teachers. The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!",
    ),
    InputExample(
        text_a = "Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.",
    ),
    InputExample(
        text_a = "This is one of the dumbest films, I've ever seen. It rips off nearly ever type of thriller and manages to make a mess of them all.<br /><br />There's not a single good line or character in the whole mess. If there was a plot, it was an afterthought and as far as acting goes, there's nothing good to say so Ill say nothing. I honestly cant understand how this type of nonsense gets produced and actually released, does somebody somewhere not at some stage think, 'Oh my god this really is a load of shite' and call it a day. Its crap like this that has people downloading illegally, the trailer looks like a completely different film, at least if you have download it, you haven't wasted your time or money Don't waste your time, this is painful.",
    ),
    InputExample(
        text_a = "Nine minutes of psychedelic, pulsating, often symmetric abstract images, are enough to drive anyone crazy. I did spot a full-frame eye at the start, and later some birds silhouetted against other colors. It was just not my cup of tea. It's about 8½ minutes too long.",
    ),
    InputExample(
        text_a = "I did not like the idea of the female turtle at all since 1987 we knew the TMNT to be four brothers with their teacher Splinter and their enemies and each one of the four brothers are named after the great artists name like Leonardo , Michelangleo, Raphel and Donatello so Venus here doesn't have any meaning or playing any important part and I believe that the old TMNT series was much more better than that new one which contains Venus As a female turtle will not add any action to the story we like the story of the TMNT we knew in 1987 to have new enemies in every part is a good point to have some action but to have a female turtle is a very weak point to have some action, we wish to see more new of TMNT series but just as the same characters we knew in 1987 without that female turtle.",
    ),
    InputExample(
        text_a = "Three part horror film with some guy in a boarded up house imploring the viewer not to go out there and (unfortunately) gives us three tales to prove why.<br /><br />The first story involves a young couple in a car accident who meet up with two psychos. It leads up to two totally predictable twists. Still, it's quick (about 15 minutes), violent, well-acted and well-done. Predictable but enjoyable.<br /><br />The second involves a man on the run after stealing a large amount of money. His car breaks down, he's attacked by a dog and stumbles into a nearby clinic. VERY obvious, badly done and extremely slow. Even at 30 minutes this is too long. Good acting though.<br /><br />The third is just barely a horror story. It involves a beautiful, lonely woman looking for Mr. Right. It has beautiful set designs, a nice erotic feel and a nice sex scene. But (again) predictable and not even remotely scary.<br /><br />It ends very stupidly.<br /><br />All in all, the first one is worth watching, but that's it. Tune in for that one then turn it off.",
    ),
    InputExample(
        text_a = "Not a `woman film' but film for the gang. One of the worst films ever made by a male director about woman. Director Andy McKay simply doesn't know woman. Peaks of bad taste, American Pie's humor style, crude story, no sense, groundless story, refuted characters. Vulgar fantasies came to life on screen. Insulting and definitely not funny. I wonder how three good actresses accepted to take part in it.",
    ),
]

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "/home/percent1/models/nlp/text-classification/pretrained/bert-base-cased")

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)

from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = tokenizer,
)

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)

from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

import torch

# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'

