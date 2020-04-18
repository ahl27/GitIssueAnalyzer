import pickle
import github
from authinfo import *
import re

def get_issues(repo_name):
    #log in
    g = github.Github(API_token)
    #get repo
    print("Logging into github...")
    repo = g.get_repo("CSSEGISandData/COVID-19")

    print("Getting issues...")
    open_issues = repo.get_issues(state="open")
    closed_issues = repo.get_issues(state="closed")

    open_issues_lst = dict()
    corpus = []
    closed_issues_lst = []

    #save body of each issue
    print("Saving content of each issue...")
    for issue in open_issues:
        issue_text = (issue.title + "\n" + issue.body).lower()

        regex = re.compile("[^a-z\ \n\/\.\(\)]")
        issue_text = regex.sub('', issue_text)
        regex = re.compile("[\.\/\(\)]")
        issue_text = regex.sub(' ', issue_text)

        open_issues_lst[issue.number] = issue_text
        corpus.append(issue_text.lower())
    with open("issues_pickled.pkl", "wb") as f:
        pickle.dump(corpus, f)
    return corpus
