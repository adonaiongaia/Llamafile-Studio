//! A small, purpose-built ZIP writer for llamafile packaging.
//!
//! # Why not the `zip` crate?
//!
//! A `.llamafile` is an APE (Actually Portable Executable) with a ZIP archive
//! appended to it. Two properties of that archive are load-bearing:
//!
//! 1. **Stored (method 0) entries.** llamafile `mmap()`s the archive and hands
//!    pointers directly to GPU backends, so the weights must exist verbatim in
//!    the file. Compression would force a decompression pass into anonymous
//!    memory and destroy the zero-copy path.
//! 2. **16 KiB aligned payloads.** Data must start on a 16384-byte boundary for
//!    those mmap'd pointers to satisfy the backend alignment requirements: that
//!    is the value llamafile's own `zipalign` uses, and anything lower makes the
//!    runtime warn and map the weights with a skewed offset.
//!
//! Upstream solves this with a bespoke `zipalign` tool (roughly 500 lines of C).
//! This module is the Rust equivalent: it computes absolute (prepended) offsets
//! while streaming, pads each local header with a dummy extra field until the
//! payload lands on a page boundary, and writes a ZIP64 directory whenever a
//! size or offset exceeds the 32-bit limit - which happens for any model above
//! 4 GiB, i.e. most 7B+ Q8 models.
//!
//! Everything is little-endian, per the ZIP specification (APPNOTE 6.3.10).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Payload alignment for archive entries, in bytes.
///
/// This is 16 KiB because that is what upstream `zipalign` writes and what the
/// runtime checks (`data_offset & 16383`) before mapping the weights. A merely
/// page-aligned payload still runs, but llamafile prints "use zipalign (rather
/// than zip) to create llamafiles" on every model open and maps the tensors at a
/// skew, which defeats the zero-copy path the archive is built for.
pub const ZIP_ALIGNMENT: u64 = 16 * 1024;

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const ZIP64_LOCATOR_SIG: u32 = 0x0706_4b50;

const EXTRA_ZIP64: u16 = 0x0001;
/// Private extra-field id used purely as alignment filler ("LZ").
const EXTRA_PADDING: u16 = 0x5a4c;

const FLAG_UTF8: u16 = 0x0800;
const METHOD_STORED: u16 = 0;
const VERSION_DEFAULT: u16 = 20;
const VERSION_ZIP64: u16 = 45;
/// 3.63 on Unix, which is what the reference implementation advertises.
const VERSION_MADE_BY: u16 = 0x033f;
/// `0100644 << 16`: regular file, rw-r--r--.
const EXTERNAL_ATTRS: u32 = 0o100644 << 16;

/// One entry after it has been written, as reported back to the UI.
#[derive(Debug, Clone)]
pub struct EntrySummary {
    pub name: String,
    pub size: u64,
    /// Absolute offset (from the start of the whole `.llamafile`) of the data.
    pub data_offset: u64,
    pub zip64: bool,
}

/// Result of finishing an archive.
#[derive(Debug, Clone)]
pub struct ZipSummary {
    pub entries: Vec<EntrySummary>,
    /// Offset of the central directory, absolute from the file start.
    pub central_directory_offset: u64,
    /// Total file size once the archive is complete.
    pub total_size: u64,
    pub uses_zip64: bool,
}

struct CentralRecord {
    name: Vec<u8>,
    crc: u32,
    size: u64,
    local_offset: u64,
    zip64: bool,
    time: u16,
    date: u16,
    data_offset: u64,
}

/// Streaming writer that appends stored entries to an already populated file.
///
/// The writer is deliberately bound to a `File` (not a generic `Write`) because
/// it seeks backwards to patch CRCs once a payload has been streamed - we never
/// want to hold a multi-gigabyte model in memory just to compute its checksum.
pub struct StoredZipWriter<'a> {
    file: &'a mut File,
    alignment: u64,
    records: Vec<CentralRecord>,
}

impl<'a> StoredZipWriter<'a> {
    /// Creates a writer that continues at the current end of `file`.
    pub fn new(file: &'a mut File, alignment: u64) -> Result<Self, String> {
        // Defensive: make sure we start appending at a sane position.
        let len = file
            .seek(SeekFrom::End(0))
            .map_err(|e| format!("cannot seek to the end of the output file: {e}"))?;
        if len == 0 {
            return Err(
                "the output file is empty: copy the llamafile runtime into it before appending entries"
                    .to_string(),
            );
        }
        Ok(Self {
            file,
            alignment: alignment.max(1),
            records: Vec::new(),
        })
    }

    /// Appends a small in-memory entry (used for `.args` and `persona.txt`).
    pub fn add_bytes(&mut self, name: &str, data: &[u8]) -> Result<EntrySummary, String> {
        let mut cursor = std::io::Cursor::new(data);
        let len = data.len() as u64;
        self.add_stream(name, &mut cursor, len, &mut |_| {})
    }

    /// Appends a file, streaming it in 1 MiB chunks so memory use stays flat.
    ///
    /// `progress` receives the number of freshly consumed bytes; the caller uses
    /// it to drive the UI progress bar.
    pub fn add_file(
        &mut self,
        name: &str,
        path: &Path,
        progress: &mut dyn FnMut(u64),
    ) -> Result<EntrySummary, String> {
        let mut source = File::open(path)
            .map_err(|e| format!("cannot open {} for reading: {e}", path.display()))?;
        let size = source
            .metadata()
            .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
            .len();
        if size == 0 {
            return Err(format!("{} is empty (0 bytes)", path.display()));
        }
        self.add_stream(name, &mut source, size, progress)
    }

    fn add_stream<R: Read>(
        &mut self,
        name: &str,
        reader: &mut R,
        size: u64,
        progress: &mut dyn FnMut(u64),
    ) -> Result<EntrySummary, String> {
        if name.is_empty() {
            return Err("zip entry name must not be empty".to_string());
        }
        // Normalise to forward slashes: ZIP mandates it and llamafile's loader
        // looks entries up by exact name.
        let normalised = name.replace('\\', "/");
        let name_bytes = normalised.as_bytes().to_vec();

        let header_offset = self
            .file
            .stream_position()
            .map_err(|e| format!("cannot read the output file position: {e}"))?;

        // Decide up front whether this entry needs ZIP64. Doing it before the
        // header is written keeps the patched header exactly the same length as
        // the placeholder.
        let needs_zip64 = size > u32::MAX as u64 || header_offset > u32::MAX as u64;
        let zip64_extra_len: u64 = if needs_zip64 { 4 + 16 } else { 0 };

        // Alignment: header (30) + name + zip64 extra + padding-field header (4)
        // must be followed by `pad` filler bytes so the payload starts on a page.
        let before_padding = 30 + name_bytes.len() as u64 + zip64_extra_len + 4;
        let modulus = (self.alignment - ((header_offset + before_padding) % self.alignment))
            % self.alignment;
        let padding = modulus;
        let extra_len = zip64_extra_len + 4 + padding;
        if extra_len > u16::MAX as u64 {
            return Err(format!(
                "the zip extra field for \"{normalised}\" would be too large ({extra_len} bytes)"
            ));
        }

        let (time, date) = dos_datetime(SystemTime::now());
        let data_offset = header_offset + 30 + name_bytes.len() as u64 + extra_len;

        // Pass 1: placeholder header + payload, hashing as we stream.
        self.write_local_header(
            &name_bytes,
            0,
            0,
            needs_zip64,
            padding as u16,
            time,
            date,
        )?;
        debug_assert_eq!(
            self.file.stream_position().map_err(|e| e.to_string())?,
            data_offset,
            "alignment calculation must match the physical header size"
        );

        let mut hasher = crc32fast::Hasher::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut written: u64 = 0;
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|e| format!("read error while packaging \"{normalised}\": {e}"))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            self.file
                .write_all(&buffer[..read])
                .map_err(|e| format!("write error while packaging \"{normalised}\": {e}"))?;
            written += read as u64;
            progress(read as u64);
        }
        if written != size {
            return Err(format!(
                "\"{normalised}\" changed size while being packaged (expected {size} bytes, read {written})"
            ));
        }
        let crc = hasher.finalize();

        // Pass 2: rewind and patch the real CRC + sizes in place.
        let end_of_entry = self
            .file
            .stream_position()
            .map_err(|e| format!("cannot read the output file position: {e}"))?;
        self.file
            .seek(SeekFrom::Start(header_offset))
            .map_err(|e| format!("cannot rewind to patch the zip header: {e}"))?;
        self.write_local_header(
            &name_bytes,
            crc,
            size,
            needs_zip64,
            padding as u16,
            time,
            date,
        )?;
        self.file
            .seek(SeekFrom::Start(end_of_entry))
            .map_err(|e| format!("cannot seek back to the end of the archive: {e}"))?;

        self.records.push(CentralRecord {
            name: name_bytes,
            crc,
            size,
            local_offset: header_offset,
            zip64: needs_zip64,
            time,
            date,
            data_offset,
        });

        Ok(EntrySummary {
            name: normalised,
            size,
            data_offset,
            zip64: needs_zip64,
        })
    }

    /// Writes a local file header. Called twice per entry with identical
    /// arguments except `crc`/`size`, which guarantees a stable byte length.
    fn write_local_header(
        &mut self,
        name: &[u8],
        crc: u32,
        size: u64,
        zip64: bool,
        padding: u16,
        time: u16,
        date: u16,
    ) -> Result<(), String> {
        let zip64_base: u16 = if zip64 { 4 + 16 } else { 0 };
        let extra_len = zip64_base + 4 + padding;

        let mut header = Vec::with_capacity(30 + name.len() + extra_len as usize);
        push_u32(
            &mut header,
            LOCAL_SIG,
        );
        push_u16(
            &mut header,
            if zip64 { VERSION_ZIP64 } else { VERSION_DEFAULT },
        );
        push_u16(&mut header, FLAG_UTF8);
        push_u16(&mut header, METHOD_STORED);
        push_u16(&mut header, time);
        push_u16(&mut header, date);
        push_u32(&mut header, crc);
        // Sizes are 0xFFFFFFFF when the real value lives in the ZIP64 field.
        push_u32(&mut header, if zip64 { u32::MAX } else { size as u32 });
        push_u32(&mut header, if zip64 { u32::MAX } else { size as u32 });
        push_u16(&mut header, name.len() as u16);
        push_u16(&mut header, extra_len);

        header.extend_from_slice(name);

        if zip64 {
            push_u16(&mut header, EXTRA_ZIP64);
            push_u16(&mut header, 16);
            push_u64(&mut header, size); // uncompressed
            push_u64(&mut header, size); // compressed (stored == identical)
        }

        // Alignment filler. A well-formed (id, size, payload) triple keeps every
        // ZIP reader we care about happy, including `unzip -l` and `bsdtar`.
        push_u16(&mut header, EXTRA_PADDING);
        push_u16(&mut header, padding);
        header.resize(header.len() + padding as usize, 0);

        debug_assert_eq!(header.len(), 30 + name.len() + extra_len as usize);
        self.file
            .write_all(&header)
            .map_err(|e| format!("write error while writing the zip header: {e}"))
    }

    /// Writes the central directory plus the end-of-central-directory records.
    pub fn finish(mut self) -> Result<ZipSummary, String> {
        if self.records.is_empty() {
            return Err("refusing to finalise an archive with no entries".to_string());
        }

        let cd_offset = self
            .file
            .stream_position()
            .map_err(|e| format!("cannot read the central directory offset: {e}"))?;

        // Move the records out so the central-directory writer can borrow `self`
        // mutably while we still read their metadata.
        let records = std::mem::take(&mut self.records);
        let mut any_zip64 = false;
        for record in &records {
            if self.write_central_entry(record)? {
                any_zip64 = true;
            }
        }

        let cd_end = self
            .file
            .stream_position()
            .map_err(|e| format!("cannot read the central directory end: {e}"))?;
        let cd_size = cd_end - cd_offset;
        let count = records.len() as u64;

        let needs_zip64_eocd = any_zip64
            || count > u16::MAX as u64
            || cd_offset > u32::MAX as u64
            || cd_size > u32::MAX as u64;

        if needs_zip64_eocd {
            let zip64_eocd_offset = cd_end;
            let mut rec = Vec::with_capacity(56);
            push_u32(&mut rec, ZIP64_EOCD_SIG);
            push_u64(&mut rec, 44); // size of the remainder of this record
            push_u16(&mut rec, VERSION_MADE_BY);
            push_u16(&mut rec, VERSION_ZIP64);
            push_u32(&mut rec, 0); // this disk
            push_u32(&mut rec, 0); // disk with the central directory
            push_u64(&mut rec, count); // entries on this disk
            push_u64(&mut rec, count); // entries in total
            push_u64(&mut rec, cd_size);
            push_u64(&mut rec, cd_offset);
            self.write_bytes(&rec)?;

            let mut locator = Vec::with_capacity(20);
            push_u32(&mut locator, ZIP64_LOCATOR_SIG);
            push_u32(&mut locator, 0); // disk holding the ZIP64 EOCD
            push_u64(&mut locator, zip64_eocd_offset);
            push_u32(&mut locator, 1); // total number of disks
            self.write_bytes(&locator)?;
        }

        // Classic EOCD, always last. Fields that do not fit carry sentinels and
        // the reader follows the ZIP64 locator instead.
        let mut eocd = Vec::with_capacity(22);
        push_u32(&mut eocd, EOCD_SIG);
        push_u16(&mut eocd, 0);
        push_u16(&mut eocd, 0);
        push_u16(&mut eocd, if count > u16::MAX as u64 { u16::MAX } else { count as u16 });
        push_u16(&mut eocd, if count > u16::MAX as u64 { u16::MAX } else { count as u16 });
        push_u32(
            &mut eocd,
            if cd_size > u32::MAX as u64 { u32::MAX } else { cd_size as u32 },
        );
        push_u32(
            &mut eocd,
            if cd_offset > u32::MAX as u64 { u32::MAX } else { cd_offset as u32 },
        );
        push_u16(&mut eocd, 0); // archive comment length
        self.write_bytes(&eocd)?;

        self.file
            .flush()
            .map_err(|e| format!("cannot flush the output file: {e}"))?;
        let total_size = self
            .file
            .metadata()
            .map_err(|e| format!("cannot stat the output file: {e}"))?
            .len();

        Ok(ZipSummary {
            entries: records
                .iter()
                .map(|r| EntrySummary {
                    name: String::from_utf8_lossy(&r.name).to_string(),
                    size: r.size,
                    data_offset: r.data_offset,
                    zip64: r.zip64,
                })
                .collect(),
            central_directory_offset: cd_offset,
            total_size,
            uses_zip64: needs_zip64_eocd,
        })
    }

    /// Returns `true` when the entry itself required ZIP64 fields.
    fn write_central_entry(&mut self, record: &CentralRecord) -> Result<bool, String> {
        let size_needs_zip64 = record.size > u32::MAX as u64;
        let offset_needs_zip64 = record.local_offset > u32::MAX as u64;
        let zip64 = size_needs_zip64 || offset_needs_zip64 || record.zip64;

        // Only the overflowing values are stored, in the mandated order.
        let mut zip64_payload: Vec<u8> = Vec::new();
        if zip64 {
            push_u64(&mut zip64_payload, record.size);
            push_u64(&mut zip64_payload, record.size);
            if offset_needs_zip64 || record.zip64 {
                push_u64(&mut zip64_payload, record.local_offset);
            }
        }
        let extra_len: u16 = if zip64 {
            (4 + zip64_payload.len()) as u16
        } else {
            0
        };

        let mut entry =
            Vec::with_capacity(46 + record.name.len() + extra_len as usize);
        push_u32(&mut entry, CENTRAL_SIG);
        push_u16(&mut entry, VERSION_MADE_BY);
        push_u16(
            &mut entry,
            if zip64 { VERSION_ZIP64 } else { VERSION_DEFAULT },
        );
        push_u16(&mut entry, FLAG_UTF8);
        push_u16(&mut entry, METHOD_STORED);
        push_u16(&mut entry, record.time);
        push_u16(&mut entry, record.date);
        push_u32(&mut entry, record.crc);
        push_u32(
            &mut entry,
            if zip64 { u32::MAX } else { record.size as u32 },
        );
        push_u32(
            &mut entry,
            if zip64 { u32::MAX } else { record.size as u32 },
        );
        push_u16(&mut entry, record.name.len() as u16);
        push_u16(&mut entry, extra_len);
        push_u16(&mut entry, 0); // comment length
        push_u16(&mut entry, 0); // disk number start
        push_u16(&mut entry, 0); // internal attributes
        push_u32(&mut entry, EXTERNAL_ATTRS);
        push_u32(
            &mut entry,
            if offset_needs_zip64 || record.zip64 {
                u32::MAX
            } else {
                record.local_offset as u32
            },
        );
        entry.extend_from_slice(&record.name);
        if zip64 {
            push_u16(&mut entry, EXTRA_ZIP64);
            push_u16(&mut entry, zip64_payload.len() as u16);
            entry.extend_from_slice(&zip64_payload);
        }

        self.write_bytes(&entry)?;
        Ok(zip64)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.file
            .write_all(bytes)
            .map_err(|e| format!("write error while finalising the archive: {e}"))
    }

}

/// Confirms that the file at `path` still ends with a readable ZIP archive.
///
/// Every llamafile runtime is an APE with a PKZIP archive in its tail, and this
/// is the check that tells a complete runtime from a truncated download: the
/// APE header survives a cut-off transfer, the archive tail does not. The scan
/// mirrors llamafile's own loader (`llamafile_open_zip`): the end-of-central-
/// directory record has to live in the last 64 KiB, and the central directory it
/// points at has to lie inside the file.
pub fn verify_trailing_archive(path: &Path) -> Result<(), String> {
    const EOCD_MIN: usize = 22;
    const ZIP64_LOCATOR_SIZE: usize = 20;
    const ZIP64_EOCD_SIZE: u64 = 56;

    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    if size < EOCD_MIN as u64 {
        return Err("the file is too small to be a llamafile runtime".to_string());
    }

    let tail_len = size.min(65_536) as usize;
    file.seek(SeekFrom::Start(size - tail_len as u64))
        .map_err(|e| format!("cannot seek within {}: {e}", path.display()))?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail)
        .map_err(|e| format!("cannot read the end of {}: {e}", path.display()))?;

    let read_u32 = |bytes: &[u8], at: usize| {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };

    for at in (0..=tail_len - EOCD_MIN).rev() {
        if read_u32(&tail, at) != EOCD_SIG {
            continue;
        }
        let mut entries = u16::from_le_bytes([tail[at + 10], tail[at + 11]]) as u64;
        let mut cd_size = read_u32(&tail, at + 12) as u64;
        let mut cd_offset = read_u32(&tail, at + 16) as u64;

        if cd_offset == u32::MAX as u64 || cd_size == u32::MAX as u64 || entries == u16::MAX as u64
        {
            // ZIP64: the real numbers live in the ZIP64 end record, which the
            // locator immediately before the classic EOCD points at.
            if at < ZIP64_LOCATOR_SIZE {
                continue;
            }
            let locator = at - ZIP64_LOCATOR_SIZE;
            if read_u32(&tail, locator) != ZIP64_LOCATOR_SIG {
                continue;
            }
            let zip64_at =
                u64::from_le_bytes(tail[locator + 8..locator + 16].try_into().unwrap());
            if zip64_at + ZIP64_EOCD_SIZE > size {
                return Err("its ZIP64 record points outside the file".to_string());
            }
            let mut record = [0u8; ZIP64_EOCD_SIZE as usize];
            file.seek(SeekFrom::Start(zip64_at))
                .map_err(|e| format!("cannot seek within {}: {e}", path.display()))?;
            file.read_exact(&mut record)
                .map_err(|e| format!("cannot read the ZIP64 record of {}: {e}", path.display()))?;
            if read_u32(&record, 0) != ZIP64_EOCD_SIG {
                continue;
            }
            entries = u64::from_le_bytes(record[32..40].try_into().unwrap());
            cd_size = u64::from_le_bytes(record[40..48].try_into().unwrap());
            cd_offset = u64::from_le_bytes(record[48..56].try_into().unwrap());
        }

        if entries == 0 || cd_size < 46 {
            return Err("its embedded archive has no entries".to_string());
        }
        match cd_offset.checked_add(cd_size) {
            Some(end) if end <= size => return Ok(()),
            _ => return Err("its embedded archive points past the end of the file".to_string()),
        }
    }

    Err("it has no embedded ZIP archive at the end of the file".to_string())
}

/* ------------------------------------------------------------ byte helpers */

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/* ------------------------------------------------------------------- dates */

/// Converts a `SystemTime` into MS-DOS (time, date) pairs, clamped to the
/// representable 1980..=2107 range. Invalid pre-epoch clocks fall back to the
/// DOS epoch, exactly like every other archiver does.
fn dos_datetime(stamp: SystemTime) -> (u16, u16) {
    let seconds = match stamp.duration_since(UNIX_EPOCH) {
        Ok(delta) => delta.as_secs() as i64,
        Err(_) => 0,
    };
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;

    let year = year.clamp(1980, 2107) as u32;
    let date = ((year - 1980) << 9) | ((month as u32) << 5) | day as u32;
    let time = ((hour as u32) << 11) | ((minute as u32) << 5) | ((second as u32) / 2);
    (time as u16, date as u16)
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 to (y, m, d).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m: u32 = if mp < 10 { (mp + 3) as u32 } else { (mp - 9) as u32 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_aligned_stored_entries() {
        let dir = std::env::temp_dir().join(format!("lms-zip-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.llamafile");

        // Pretend 1000 bytes of APE binary.
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0x41u8; 1000]).unwrap();
        drop(f);

        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::End(0)).unwrap();
        let mut zip = StoredZipWriter::new(&mut f, ZIP_ALIGNMENT).unwrap();
        zip.add_bytes(".args", b"-m\n/zip/m.gguf\n...\n").unwrap();
        // A payload big enough to span several pages, like a real model.
        zip.add_bytes("fake.gguf", &vec![0x5au8; 9000]).unwrap();
        let summary = zip.finish().unwrap();

        assert_eq!(summary.entries.len(), 2);
        for entry in &summary.entries {
            assert_eq!(entry.data_offset % ZIP_ALIGNMENT, 0, "{} misaligned", entry.name);
        }
        // Entries must be laid out back to back in write order.
        assert!(summary.entries[1].data_offset > summary.entries[0].data_offset);
        assert!(summary.total_size > 1000);
        // What llamafile checks before mapping the weights.
        assert_eq!(summary.entries[1].data_offset & 16383, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lms-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn accepts_a_runtime_whose_archive_survived() {
        let dir = scratch_dir("tail-ok");
        let path = dir.join("runtime");
        // Pretend 1000 bytes of APE, then a real archive in the tail.
        let mut f = File::create(&path).unwrap();
        f.write_all(&vec![0x4du8; 1000]).unwrap();
        drop(f);

        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::End(0)).unwrap();
        let mut zip = StoredZipWriter::new(&mut f, ZIP_ALIGNMENT).unwrap();
        zip.add_bytes(".cosmo", b"stub").unwrap();
        zip.finish().unwrap();
        drop(f);

        assert!(verify_trailing_archive(&path).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_truncated_runtime() {
        let dir = scratch_dir("tail-truncated");
        let path = dir.join("runtime.part");
        // An oversized APE header whose archive tail never arrived.
        std::fs::write(&path, vec![0x4du8; 100_000]).unwrap();
        let error = verify_trailing_archive(&path).unwrap_err();
        assert!(error.contains("no embedded ZIP archive"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn accepts_a_zip64_archive_tail() {
        // Mirrors a real full runtime, whose own archive needs ZIP64.
        let dir = scratch_dir("tail-zip64");
        let path = dir.join("runtime");
        let mut bytes = vec![0x4du8; 512];
        let zip64_at = bytes.len() as u64;
        push_u32(&mut bytes, ZIP64_EOCD_SIG);
        push_u64(&mut bytes, 44);
        push_u16(&mut bytes, VERSION_MADE_BY);
        push_u16(&mut bytes, VERSION_ZIP64);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 46);
        push_u64(&mut bytes, 0);
        push_u32(&mut bytes, ZIP64_LOCATOR_SIG);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, zip64_at);
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, EOCD_SIG);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, u16::MAX);
        push_u16(&mut bytes, u16::MAX);
        push_u32(&mut bytes, u32::MAX);
        push_u32(&mut bytes, u32::MAX);
        push_u16(&mut bytes, 0);
        std::fs::write(&path, &bytes).unwrap();

        assert!(verify_trailing_archive(&path).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dos_dates_are_stable() {
        let (time, date) = dos_datetime(UNIX_EPOCH);
        // 1980-01-01: ((1980 - 1980) << 9) | (1 << 5) | 1 == 33.
        assert_eq!(date, 33);
        assert_eq!(time, 0);
    }
}
