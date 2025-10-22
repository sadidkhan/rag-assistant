issue = {
  "key": "PROJ-123",
  "url": "https://your-jira/browse/PROJ-123",
  "project": "PROJ",
  "title": fields["summary"],
  "type": fields["issuetype"]["name"],           # Bug/Story/Task
  "status": fields["status"]["name"],
  "priority": (fields.get("priority") or {}).get("name"),
  "assignee": (fields.get("assignee") or {}).get("displayName"),
  "labels": fields.get("labels", []),
  "components": [c["name"] for c in fields.get("components", [])],
  "sprint": "...",                               # if you use Jira Software
  "created": fields["created"],
  "updated": fields["updated"],
  "description": fields.get("description") or "",
  "comments": [
      {"id": c["id"], "author": c["author"]["displayName"],
       "created": c["created"], "body": c["body"]}
      for c in fields.get("comment", {}).get("comments", [])
  ],
  # Optional parsed sections if you have them:
  "steps_to_reproduce": "...",
  "expected_result": "...",
  "actual_result": "...",
  "acceptance_criteria": "..."
}


from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_header_part(issue):
    header_text = (
        f"[{issue['key']}] {issue['title']}\n"
        f"Type: {issue['type']} | Status: {issue['status']} | Priority: {issue.get('priority')}\n"
        f"Assignee: {issue.get('assignee')} | Labels: {', '.join(issue.get('labels', []))}\n"
        f"Components: {', '.join(issue.get('components', []))}\n"
    )
    docs = [Document(page_content=header_text, metadata={
        "source": "jira", "section": "header",
        "issue_key": issue["key"], "issue_url": issue["url"], "project": issue["project"],
        "title": issue["title"], "type": issue["type"], "status": issue["status"],
        "priority": issue.get("priority"), "assignee": issue.get("assignee"),
        "labels": issue.get("labels", []), "created_at": issue["created"], "updated_at": issue["updated"],
    })]
    return docs

def chunk_description_part(issue):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,          # Jira default: 2000 chars
        chunk_overlap=200,        # ~10%
        add_start_index=True,
        separators=["\n\n", "\n- ", "\n* ", "\n", " ", ""],  # keep lists/paras intact
    )

    def add_section(text, name):
        if not text or not text.strip(): return []
        chunks = splitter.create_documents([text])
        for d in chunks:
            d.metadata.update({ "source": "jira", "section": name,
                "issue_key": issue["key"], "issue_url": issue["url"], "project": issue["project"],
                "title": issue["title"], "type": issue["type"], "status": issue["status"]
            })
        return chunks

    docs = []
    docs += add_section(issue.get("description",""), "description")
    docs += add_section(issue.get("steps_to_reproduce",""), "steps")
    docs += add_section(issue.get("expected_result",""), "expected")
    docs += add_section(issue.get("actual_result",""), "actual")
    docs += add_section(issue.get("acceptance_criteria",""), "acceptance_criteria")


def chunk_comments_part(issue):
    for c in issue.get("comments", []):
        body = c["body"] or ""
        if len(body) <= 1800:
            docs.append(Document(
                page_content=body,
                metadata={
                "source":"jira","section":"comment","issue_key":issue["key"],"issue_url":issue["url"],
                "comment_id": c["id"], "author": c["author"], "created_at": c["created"]
                }
            ))
    else:
        for d in splitter.create_documents([body]):
            d.metadata.update({
              "source":"jira","section":"comment","issue_key":issue["key"],
              "issue_url":issue["url"],"comment_id": c["id"], "author": c["author"],
              "created_at": c["created"]
            })
            docs.append(d)

