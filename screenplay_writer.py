import re
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

# Use Path for file locations
current_dir = Path.cwd()
agents_config_path = current_dir / "config" / "agents.yaml"
tasks_config_path = current_dir / "config" / "tasks.yaml"

# Load YAML configuration files
with open(agents_config_path, "r") as file:
    agents_config = yaml.safe_load(file)

with open(tasks_config_path, "r") as file:
    tasks_config = yaml.safe_load(file)

## Define Agents
spamfilter = Agent(
    config=agents_config["spamfilter"], allow_delegation=False, verbose=True
)

analyst = Agent(config=agents_config["analyst"], allow_delegation=False, verbose=True)

scriptwriter = Agent(
    config=agents_config["scriptwriter"], allow_delegation=False, verbose=True
)

formatter = Agent(
    config=agents_config["formatter"], allow_delegation=False, verbose=True
)


scorer = Agent(config=agents_config["scorer"], allow_delegation=False, verbose=True)


# this is one example of a public post in the newsgroup alt.atheism
# try it out yourself by replacing this with your own email thread or text or ...
discussion = """                                     USA

FREEDOM FROM RELIGION FOUNDATION

Darwin fish bumper stickers and assorted other atheist paraphernalia are
available from the Freedom From Religion Foundation in the US.

Write to:  FFRF, P.O. Box 750, Madison, WI 53701.
Telephone: (608) 256-8900

EVOLUTION DESIGNS

Evolution Designs sell the "Darwin fish".  It's a fish symbol, like the ones
Christians stick on their cars, but with feet and the word "Darwin" written
inside.  The deluxe moulded 3D plastic fish is $4.95 postpaid in the US.

Write to:  Evolution Designs, 7119 Laurel Canyon #4, North Hollywood,
           CA 91605.

People in the San Francisco Bay area can get Darwin Fish from Lynn Gold --
try mailing <figmo@netcom.com>.  For net people who go to Lynn directly, the
price is $4.95 per fish.

AMERICAN ATHEIST PRESS

AAP publish various atheist books -- critiques of the Bible, lists of
Biblical contradictions, and so on.  One such book is:

"The Bible Handbook" by W.P. Ball and G.W. Foote.  American Atheist Press.
372 pp.  ISBN 0-910309-26-4, 2nd edition, 1986.  Bible contradictions,
absurdities, atrocities, immoralities... contains Ball, Foote: "The Bible
Contradicts Itself", AAP.  Based on the King James version of the Bible.

Write to:  American Atheist Press, P.O. Box 140195, Austin, TX 78714-0195.
      or:  7215 Cameron Road, Austin, TX 78752-2973.
Telephone: (512) 458-1244
Fax:       (512) 467-9525

PROMETHEUS BOOKS

Sell books including Haught's "Holy Horrors" (see below).

Write to:  700 East Amherst Street, Buffalo, New York 14215.
Telephone: (716) 837-2475.

An alternate address (which may be newer or older) is:
Prometheus Books, 59 Glenn Drive, Buffalo, NY 14228-2197.

AFRICAN-AMERICANS FOR HUMANISM

An organization promoting black secular humanism and uncovering the history of
black freethought.  They publish a quarterly newsletter, AAH EXAMINER.

Write to:  Norm R. Allen, Jr., African Americans for Humanism, P.O. Box 664,
           Buffalo, NY 14226



"""

# Filter out spam and vulgar posts
task0 = Task(
    description=tasks_config["task0"]["description"],
    expected_output=tasks_config["task0"]["expected_output"],
    agent=spamfilter,
)
result = task0.execute()
if "STOP" in result:
    # stop here and proceed to next post
    print("This spam message will be filtered out")

# process post with a crew of agents, ultimately delivering a well formatted dialogue
task1 = Task(
    description=tasks_config["task1"]["description"],
    expected_output=tasks_config["task1"]["expected_output"],
    agent=analyst,
)

task2 = Task(
    description=tasks_config["task2"]["description"],
    expected_output=tasks_config["task2"]["expected_output"],
    agent=scriptwriter,
)

task3 = Task(
    description=tasks_config["task3"]["description"],
    expected_output=tasks_config["task3"]["expected_output"],
    agent=formatter,
)
crew = Crew(
    agents=[analyst, scriptwriter, formatter],
    tasks=[task1, task2, task3],
    verbose=2,  # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = crew.kickoff()

# get rid of directions and actions between brackets, eg: (smiling)
result = re.sub(r"\(.*?\)", "", result)

print("===================== end result from crew ===================================")
print(result)
print("===================== score ==================================================")
task4 = Task(
    description=tasks_config["task4"]["description"],
    expected_output=tasks_config["task4"]["expected_output"],
    agent=scorer,
)

score = task4.execute()
score = score.split("\n")[0]  # sometimes an explanation comes after score, ignore
print(f"Scoring the dialogue as: {score}/10")