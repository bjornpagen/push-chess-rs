//! Local read-only control socket. The tournament remains the single database
//! owner; status/report answers come from its committed bumbledb snapshot.
use super::{Corpus, Result};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Read, Write};
use std::os::unix::{
    ffi::OsStrExt,
    fs::PermissionsExt,
    net::{UnixListener, UnixStream},
};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn endpoint(db: &Path) -> Result<PathBuf> {
    // macOS sockaddr_un cannot fit many perfectly valid database paths.
    // Canonical identity also makes relative paths and symlinks agree. Keep
    // the full digest, and never unlink an existing endpoint to claim it.
    let canonical = db.canonicalize()?;
    let digest = Sha256::digest(canonical.as_os_str().as_bytes());
    Ok(PathBuf::from(format!(
        "/tmp/push-chess-lab-{digest:x}.sock"
    )))
}
pub struct Control {
    listener: UnixListener,
    path: PathBuf,
}
impl Control {
    pub fn bind(db: &Path) -> Result<Self> {
        let path = endpoint(db)?;
        let listener = UnixListener::bind(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        Ok(Self { listener, path })
    }
    pub fn serve(&self, corpus: &Corpus) {
        // One client per writer tick bounds observation overhead.
        let Ok((mut stream, _)) = self.listener.accept() else {
            return;
        };
        let result = (|| -> Result<serde_json::Value> {
            stream.set_read_timeout(Some(Duration::from_millis(100)))?;
            stream.set_write_timeout(Some(Duration::from_millis(500)))?;
            let mut line = String::new();
            std::io::BufReader::new((&mut stream).take(1024)).read_line(&mut line)?;
            let command: serde_json::Value = serde_json::from_str(&line)?;
            match command["command"].as_str() {
                Some("summary") => corpus.summary(),
                Some("report") => corpus.report(command["run"].as_u64().ok_or("missing run")?),
                _ => Err("unknown read-only command".into()),
            }
        })();
        let response = match result {
            Ok(value) => value,
            Err(e) => serde_json::json!({"error":e.to_string()}),
        };
        let _ = writeln!(stream, "{response}");
    }
}
impl Drop for Control {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
pub fn status(db: &Path, run: Option<u64>) -> Result<serde_json::Value> {
    let mut stream = UnixStream::connect(endpoint(db)?)?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    writeln!(
        stream,
        "{}",
        serde_json::json!({"command":if run.is_some(){"report"}else{"summary"},"run":run})
    )?;
    let mut line = String::new();
    std::io::BufReader::new(stream.take(16 << 20)).read_line(&mut line)?;
    let value: serde_json::Value = serde_json::from_str(&line)?;
    if let Some(error) = value.get("error") {
        return Err(format!("database status: {error}").into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_paths_and_aliases_share_a_short_exclusive_endpoint() {
        let root = tempfile::tempdir().unwrap();
        let db = root.path().join("a".repeat(120));
        std::fs::create_dir(&db).unwrap();
        let alias = root.path().join("alias");
        std::os::unix::fs::symlink(&db, &alias).unwrap();
        assert_eq!(endpoint(&db).unwrap(), endpoint(&alias).unwrap());
        let control = Control::bind(&db).unwrap();
        assert!(control.path.as_os_str().as_bytes().len() < 104);
        assert_eq!(
            std::fs::metadata(&control.path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(Control::bind(&alias).is_err());
        assert!(UnixStream::connect(endpoint(&alias).unwrap()).is_ok());
        let path = control.path.clone();
        drop(control);
        assert!(!path.exists());
    }
}
