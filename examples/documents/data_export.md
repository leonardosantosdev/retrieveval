# Exporting your data

Every workspace can be exported in full. Go to **Account → Data → Export** and
start an export job. Large workspaces take a while, so the job runs in the
background and you receive an email with a download link when it finishes.

## Format

Exports are ZIP archives containing newline-delimited JSON, one file per
resource type, plus any uploaded attachments in their original format. The
schema matches the public API, so an export can be replayed through the API
to populate another workspace.

## Retention of export files

Download links expire after 7 days. Generating a new export is free and can be
done as often as you like.

## Scheduled exports

Scheduled nightly exports to your own object storage bucket are available on
the Scale plan. Configure the destination bucket and credentials under
**Account → Data → Scheduled export**.
