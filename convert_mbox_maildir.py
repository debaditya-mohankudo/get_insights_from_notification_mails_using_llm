import mailbox
import os
from pathlib import Path

def convert_mbox_to_maildir(mbox_path: str, dest_dir: str):
    dest = Path(dest_dir)

    # Ensure valid Maildir skeleton
    (dest / "cur").mkdir(parents=True, exist_ok=True)
    (dest / "new").mkdir(exist_ok=True)
    (dest / "tmp").mkdir(exist_ok=True)

    mbox = mailbox.mbox(mbox_path)
    maildir = mailbox.Maildir(dest_dir, create=True)

    for key, msg in mbox.iteritems():
        maildir.add(msg)

    print(f"Converted {len(mbox)} messages → {dest_dir}")


if __name__ == "__main__":
    ROOT = Path("/Users/debaditya/workspace/trash_mails")
    MAILDIR_ROOT = Path("/Users/debaditya/workspace/trash_mails_maildir")

    MAILDIR_ROOT.mkdir(exist_ok=True)

    for item in ROOT.iterdir():
        if item.suffix == ".mbox":
            mbox_path = item / "mbox"
            if mbox_path.exists():
                dest_maildir = MAILDIR_ROOT / f"{item.stem}_maildir"
                convert_mbox_to_maildir(str(mbox_path), str(dest_maildir))