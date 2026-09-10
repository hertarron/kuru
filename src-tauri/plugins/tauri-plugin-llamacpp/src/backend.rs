use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ============================================================================
// Upstream release assets
// ============================================================================

/// Where the binaries come from. ggml-org publishes a build per commit, several
/// times a day; janhq's mirror stopped at b9967 on 2026-07-22.
const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
const RELEASE_DOWNLOAD: &str = "https://github.com/ggml-org/llama.cpp/releases/download";

/// A release asset's backend token, split into the parts that decide whether
/// this machine can run it. The token IS the backend id kuru stores, so
/// `win-cuda-12.4-x64` names both the asset and the install directory.
///
/// Shapes, all `{os}[-{family}[-{variant}]]-{arch}`:
///   win-cpu-x64, win-cuda-12.4-x64, win-vulkan-x64, win-rocm-10.0-x64,
///   ubuntu-x64, ubuntu-vulkan-arm64, ubuntu-sycl-fp16-x64, macos-arm64
#[derive(Debug, PartialEq)]
pub struct BackendToken {
    pub os: String,
    pub arch: String,
    /// cpu, cuda, vulkan, rocm, sycl, opencl, openvino
    pub family: String,
    /// CUDA major, for picking the matching cudart redistributable.
    pub cuda_major: Option<u32>,
}

/// Families kuru will install. Everything else upstream publishes (sycl,
/// openvino, opencl, s390x) is left alone rather than half-supported.
const KNOWN_FAMILIES: [&str; 4] = ["cuda", "vulkan", "rocm", "cpu"];

pub fn parse_backend_token(token: &str) -> Option<BackendToken> {
    let parts: Vec<&str> = token.split('-').collect();
    if parts.len() < 2 {
        return None;
    }
    let os = parts[0].to_string();
    let arch = parts[parts.len() - 1].to_string();
    let middle = &parts[1..parts.len() - 1];

    // `ubuntu-x64` and `macos-arm64` carry no family segment and are CPU builds
    // (macOS ships Metal inside the same archive).
    let family = middle.first().copied().unwrap_or("cpu").to_string();
    if !KNOWN_FAMILIES.contains(&family.as_str()) {
        return None;
    }

    let cuda_major = if family == "cuda" {
        middle
            .get(1)
            .and_then(|v| v.split('.').next())
            .and_then(|maj| maj.parse::<u32>().ok())
    } else {
        None
    };

    Some(BackendToken {
        os,
        arch,
        family,
        cuda_major,
    })
}

/// The key `determine_supported_backends` emits and the asset filter matches
/// on. Deliberately not the full token: upstream moves the CUDA minor
/// (12.4 -> 12.6) and the ROCm version between releases, and pinning those
/// here is what makes a version list go stale. The CUDA *major* does belong in
/// the key -- a driver new enough for 12 need not be new enough for 13.
fn capability_key(os: &str, arch: &str, family: &str, cuda_major: Option<u32>) -> String {
    match cuda_major {
        Some(major) => format!("{}-{}/{}{}", os, arch, family, major),
        None => format!("{}-{}/{}", os, arch, family),
    }
}

/// Kept as the identity because kuru has no released installs to migrate.
/// It stays a command so the extension's call sites do not need to branch.
#[tauri::command]
pub fn map_old_backend_to_new(old_backend: String) -> String {
    old_backend
}

#[derive(Serialize, Deserialize)]
pub struct InstalledBackend {
    version: String,
    backend: String,
}

#[tauri::command]
pub async fn get_local_installed_backends(
    backends_dir: String,
) -> Result<Vec<InstalledBackend>, String> {
    let mut local: Vec<InstalledBackend> = Vec::new();
    let backends_path = PathBuf::from(&backends_dir);

    // Check if backends directory exists
    if !backends_path.exists() {
        return Ok(local);
    }

    // Read version directories
    let version_dirs = fs::read_dir(&backends_path)
        .map_err(|e| format!("Failed to read backends directory: {}", e))?;

    for version_entry in version_dirs {
        let version_entry =
            version_entry.map_err(|e| format!("Failed to read version entry: {}", e))?;

        let version_path = version_entry.path();

        // Check if it's a directory
        let metadata =
            fs::metadata(&version_path).map_err(|e| format!("Failed to get metadata: {}", e))?;

        if !metadata.is_dir() {
            continue;
        }

        // Get version name from path
        let version_name = match version_path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        // Read backend types in this version directory
        let backend_types = fs::read_dir(&version_path)
            .map_err(|e| format!("Failed to read version directory: {}", e))?;

        for backend_entry in backend_types {
            let backend_entry =
                backend_entry.map_err(|e| format!("Failed to read backend entry: {}", e))?;

            let backend_path = backend_entry.path();

            // Get backend name from path
            let backend_name = match backend_path.file_name() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };

            // Check if backend is actually installed
            if is_backend_installed(&backend_path) {
                local.push(InstalledBackend {
                    version: version_name.clone(),
                    backend: backend_name,
                });
            }
        }
    }

    Ok(local)
}

/// Helper function to check if a backend is properly installed
/// Checks for the existence of llama-server executable in the expected locations
fn is_backend_installed(backend_dir: &Path) -> bool {
    if !backend_dir.exists() || !backend_dir.is_dir() {
        return false;
    }

    // Determine executable name based on platform
    let exe_name = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    // First check if build directory exists (build/bin/llama-server)
    let build_path = backend_dir.join("build").join("bin").join(exe_name);
    if build_path.exists() {
        return true;
    }

    // Otherwise check root directory (llama-server)
    let root_path = backend_dir.join(exe_name);
    root_path.exists()
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BackendInfo {
    version: String,
    backend: String,
}

#[derive(Deserialize)]
pub struct SystemFeatures {
    cuda12: bool,
    cuda13: bool,
    vulkan: bool,
    #[serde(default)]
    hip: bool,
}

#[derive(Serialize)]
#[allow(dead_code)]
pub struct SupportedBackendsResult {
    supported_backend_names: Vec<String>,
    merged_backends: Vec<BackendInfo>,
}

#[tauri::command]
pub fn determine_supported_backends(
    os_type: String,
    arch: String,
    features: SystemFeatures,
) -> Result<Vec<String>, String> {
    // Upstream's own names for the platform half of an asset token.
    let (os, upstream_arch) = match (os_type.as_str(), arch.as_str()) {
        ("windows", "x86_64") | ("windows", "x86") => ("win", "x64"),
        ("windows", "aarch64") | ("windows", "arm64") => ("win", "arm64"),
        ("linux", "x86_64") | ("linux", "x86") => ("ubuntu", "x64"),
        ("linux", "aarch64") | ("linux", "arm64") => ("ubuntu", "arm64"),
        ("macos", "x86_64") | ("macos", "x86") => ("macos", "x64"),
        ("macos", "aarch64") | ("macos", "arm64") => ("macos", "arm64"),
        _ => return Err(format!("Unsupported system type: {}-{}", os_type, arch)),
    };

    let mut keys = vec![capability_key(os, upstream_arch, "cpu", None)];

    // macOS gets Metal inside the CPU archive and has no separate GPU build.
    if os != "macos" {
        if features.cuda12 {
            keys.push(capability_key(os, upstream_arch, "cuda", Some(12)));
        }
        if features.cuda13 {
            keys.push(capability_key(os, upstream_arch, "cuda", Some(13)));
        }
        if features.vulkan {
            keys.push(capability_key(os, upstream_arch, "vulkan", None));
        }
        if features.hip {
            keys.push(capability_key(os, upstream_arch, "rocm", None));
        }
    }

    Ok(keys)
}

#[tauri::command]
pub async fn list_supported_backends(
    remote_backend_versions: Vec<BackendInfo>,
    local_backend_versions: Vec<BackendInfo>,
) -> Result<Vec<BackendInfo>, String> {
    // Merge remote and local backend versions with deduplication
    let mut merged_map: HashMap<String, BackendInfo> = HashMap::new();

    for entry in remote_backend_versions {
        let key = format!("{}|{}", entry.version, entry.backend);
        merged_map.insert(key, entry);
    }

    for entry in local_backend_versions {
        let key = format!("{}|{}", entry.version, entry.backend);
        merged_map.insert(key, entry);
    }

    // Convert to vector and sort
    let mut merged: Vec<BackendInfo> = merged_map.into_values().collect();

    // Sort newest version first; if versions tie, sort by backend name
    merged.sort_by(|a, b| {
        let version_cmp = b.version.cmp(&a.version);
        if version_cmp == std::cmp::Ordering::Equal {
            a.backend.cmp(&b.backend)
        } else {
            version_cmp
        }
    });

    Ok(merged)
}

#[derive(Serialize, Deserialize)]
pub struct SupportedFeatures {
    avx: bool,
    avx2: bool,
    avx512: bool,
    cuda12: bool,
    cuda13: bool,
    vulkan: bool,
    hip: bool,
}

#[derive(Deserialize)]
pub struct GpuInfo {
    driver_version: String,
    #[serde(default)]
    vendor: Option<String>,
    nvidia_info: Option<NvidiaInfo>,
    vulkan_info: Option<VulkanInfo>,
}

#[derive(Deserialize)]
pub struct NvidiaInfo {
    #[allow(dead_code)]
    compute_capability: String,
}

#[derive(Deserialize)]
pub struct VulkanInfo {
    #[allow(dead_code)]
    api_version: String,
}

#[tauri::command]
pub fn get_supported_features(
    os_type: String,
    cpu_extensions: Vec<String>,
    gpus: Vec<GpuInfo>,
) -> Result<SupportedFeatures, String> {
    let mut features = SupportedFeatures {
        avx: cpu_extensions.contains(&"avx".to_string()),
        avx2: cpu_extensions.contains(&"avx2".to_string()),
        avx512: cpu_extensions.contains(&"avx512".to_string()),
        cuda12: false,
        cuda13: false,
        vulkan: false,
        hip: false,
    };

    // https://docs.nvidia.com/deploy/cuda-compatibility/#cuda-11-and-later-defaults-to-minor-version-compatibility
    let (min_cuda12_driver, min_cuda13_driver) = match os_type.as_str() {
        "linux" => ("525.60.13", "580"),
        "windows" => ("527.41", "580"),
        _ => return Ok(features), // Other OS types support neither CUDA nor HIP
    };

    // Check GPU features
    for gpu_info in gpus {
        let driver_version = &gpu_info.driver_version;

        // HIP (ROCm) is offered whenever an AMD GPU is present; missing ROCm
        // runtime surfaces later at install/verify/launch.
        if gpu_info.vendor.as_deref() == Some("AMD") {
            features.hip = true;
        }

        // Check CUDA support
        if gpu_info.nvidia_info.is_some() {
            if compare_versions(driver_version, min_cuda12_driver) >= 0 {
                features.cuda12 = true;
            }
            if compare_versions(driver_version, min_cuda13_driver) >= 0 {
                features.cuda13 = true;
            }
        }

        // Check Vulkan support
        if gpu_info.vulkan_info.is_some() {
            features.vulkan = true;
        }
    }

    Ok(features)
}

/// Compare version strings
/// Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
fn compare_versions(v1: &str, v2: &str) -> i32 {
    let parts1: Vec<&str> = v1.split('.').collect();
    let parts2: Vec<&str> = v2.split('.').collect();

    let max_len = parts1.len().max(parts2.len());

    for i in 0..max_len {
        let num1 = parts1
            .get(i)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let num2 = parts2
            .get(i)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        match num1.cmp(&num2) {
            std::cmp::Ordering::Less => return -1,
            std::cmp::Ordering::Greater => return 1,
            std::cmp::Ordering::Equal => continue,
        }
    }

    0
}

#[tauri::command]
pub async fn is_cuda_installed(
    backend_dir: String,
    version: String,
    os_type: String,
    jan_data_folder_path: String,
) -> Result<bool, String> {
    // Define library name lookup table
    let mut libname_lookup: HashMap<String, &str> = HashMap::new();
    libname_lookup.insert("windows-11.7".to_string(), "cudart64_110.dll");
    libname_lookup.insert("windows-12.0".to_string(), "cudart64_12.dll");
    libname_lookup.insert("windows-13.0".to_string(), "cudart64_13.dll");
    libname_lookup.insert("linux-11.7".to_string(), "libcudart.so.11.0");
    libname_lookup.insert("linux-12.0".to_string(), "libcudart.so.12");
    libname_lookup.insert("linux-13.0".to_string(), "libcudart.so.13");

    let key = format!("{}-{}", os_type, version);

    // Check if the OS-version combination is supported
    let libname = match libname_lookup.get(&key) {
        Some(name) => *name,
        None => return Ok(false),
    };

    // Expected new location: backend_dir/build/bin/libname
    let new_path = std::path::PathBuf::from(&backend_dir)
        .join("build")
        .join("bin")
        .join(libname);

    if new_path.exists() {
        return Ok(true);
    }

    // Old location (used by older builds): jan_data_folder_path/llamacpp/lib/libname
    let old_path = std::path::PathBuf::from(&jan_data_folder_path)
        .join("llamacpp")
        .join("lib")
        .join(libname);

    if old_path.exists() {
        // Ensure target directory exists
        let target_dir = PathBuf::from(&backend_dir).join("build").join("bin");

        if !target_dir.exists() {
            fs::create_dir_all(&target_dir)
                .map_err(|e| format!("Failed to create target directory: {}", e))?;
        }

        // Move old lib to the correct new location
        match fs::rename(&old_path, &new_path) {
            Ok(_) => {
                log::info!("[CUDA] Migrated {} from old path to new location.", libname);
                return Ok(true);
            }
            Err(err) => {
                log::warn!("[CUDA] Failed to move old library: {}", err);
                // Return false since the migration failed
                return Ok(false);
            }
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BestBackendResult {
    pub backend_string: String,
    pub version: String,
    pub backend_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCheckResult {
    pub update_needed: bool,
    pub new_version: String,
    pub target_backend: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[allow(dead_code)]
pub struct BackendConfigResult {
    pub best_available: String,
    pub effective_backend: String,
    pub backend_downloaded: bool,
    pub settings_updated: bool,
}

#[tauri::command]
pub fn find_latest_version_for_backend(
    version_backends: Vec<BackendInfo>,
    backend_type: String,
) -> Option<String> {
    let mut matching_backends: Vec<BackendInfo> = version_backends
        .into_iter()
        .filter(|vb| map_old_backend_to_new(vb.backend.clone()) == backend_type)
        .collect();

    if matching_backends.is_empty() {
        return None;
    }

    // Sort by version (newest first)
    matching_backends.sort_by(|a, b| b.version.cmp(&a.version));

    // Return the full string including the original asset name
    Some(format!(
        "{}/{}",
        matching_backends[0].version, matching_backends[0].backend
    ))
}

#[tauri::command]
pub async fn prioritize_backends(
    version_backends: Vec<BackendInfo>,
    has_enough_gpu_memory: bool,
) -> Result<BestBackendResult, String> {
    if version_backends.is_empty() {
        return Err("No backends available".to_string());
    }

    // Priority list based on GPU memory
    let backend_priorities: Vec<&str> = if has_enough_gpu_memory {
        vec![
            "cuda-cu13.0",
            "cuda-cu12.0",
            "cuda-cu11.7",
            "hip",
            "vulkan",
            "common_cpus",
            "avx512",
            "avx2",
            "avx",
            "noavx",
            "arm64",
            "x64",
        ]
    } else {
        vec![
            "common_cpus",
            "avx512",
            "avx2",
            "avx",
            "noavx",
            "arm64",
            "x64",
            "hip",
            "vulkan",
        ]
    };

    // Find best matching backend
    for priority_category in backend_priorities {
        let matching_backends: Vec<&BackendInfo> = version_backends
            .iter()
            .filter(|vb| {
                let category = get_backend_category(&vb.backend);
                category.as_deref() == Some(priority_category)
            })
            .collect();

        if !matching_backends.is_empty() {
            let best = matching_backends[0];
            log::info!(
                "Determined best available backend: {}/{} (Category: \"{}\")",
                best.version,
                best.backend,
                priority_category
            );

            return Ok(BestBackendResult {
                backend_string: format!("{}/{}", best.version, best.backend),
                version: best.version.clone(),
                backend_type: best.backend.clone(),
            });
        }
    }

    // Fallback to newest version
    let fallback = &version_backends[0];
    log::info!("Fallback to: {}/{}", fallback.version, fallback.backend);

    Ok(BestBackendResult {
        backend_string: format!("{}/{}", fallback.version, fallback.backend),
        version: fallback.version.clone(),
        backend_type: fallback.backend.clone(),
    })
}

/// Groups an install with the other builds of the same hardware family, so an
/// update check follows `win-cuda-12.4-x64` to `win-cuda-12.6-x64` when
/// upstream moves the minor. CUDA keeps its major because 12 and 13 need
/// different drivers and different cudart.
fn get_backend_category(backend_string: &str) -> Option<String> {
    let token = parse_backend_token(backend_string)?;
    Some(match (token.family.as_str(), token.cuda_major) {
        ("cuda", Some(major)) => format!("cuda-{}", major),
        (family, _) => family.to_string(),
    })
}

#[tauri::command]
pub fn parse_backend_version(version_string: String) -> u32 {
    // Remove any leading non-digit characters
    let numeric = version_string.trim_start_matches(|c: char| !c.is_ascii_digit());
    numeric.parse::<u32>().unwrap_or(0)
}

#[tauri::command]
pub async fn check_backend_for_updates(
    current_backend_string: String,
    version_backends: Vec<BackendInfo>,
) -> Result<UpdateCheckResult, String> {
    let parts: Vec<&str> = current_backend_string.split('/').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid current backend format: {}",
            current_backend_string
        ));
    }

    let current_version = parts[0];
    let current_backend = parts[1];

    // Get the effective/migrated backend type
    let current_effective_backend_type = map_old_backend_to_new(current_backend.to_string());

    // Find the latest version for the current backend type
    let target_backend_string =
        find_latest_version_for_backend(version_backends, current_effective_backend_type.clone());

    if target_backend_string.is_none() {
        log::warn!(
            "No available versions found for current backend type: {}",
            current_effective_backend_type
        );
        return Ok(UpdateCheckResult {
            update_needed: false,
            new_version: "0".to_string(),
            target_backend: None,
        });
    }

    let target_backend_string = target_backend_string.unwrap();
    let target_parts: Vec<&str> = target_backend_string.split('/').collect();
    let latest_version = target_parts[0];

    // Check if update is needed
    if parse_backend_version(latest_version.to_string())
        > parse_backend_version(current_version.to_string())
    {
        log::info!(
            "New update available: {} -> {}",
            latest_version,
            target_backend_string
        );
        Ok(UpdateCheckResult {
            update_needed: true,
            new_version: latest_version.to_string(),
            target_backend: Some(target_backend_string),
        })
    } else {
        log::info!(
            "Already at latest version: {} = {}",
            current_version,
            latest_version
        );
        Ok(UpdateCheckResult {
            update_needed: false,
            new_version: "0".to_string(),
            target_backend: None,
        })
    }
}

/// Prune installs of `backend_type`, always keeping `latest_version` and the
/// `keep_previous` newest versions below it. Those retained copies are what
/// makes a rollback possible without a re-download, so this must never be run
/// before a switch has proven itself.
///
/// Ordering is by parsed build number, not directory name: `b9100` must sort
/// above `b982`, which a lexical compare gets backwards.
#[tauri::command]
pub async fn remove_old_backend_versions(
    backends_dir: String,
    latest_version: String,
    backend_type: String,
    keep_previous: usize,
) -> Result<Vec<String>, String> {
    let mut removed_paths = Vec::new();
    let backends_path = PathBuf::from(&backends_dir);

    if !backends_path.exists() {
        return Ok(removed_paths);
    }

    let version_dirs = fs::read_dir(&backends_path)
        .map_err(|e| format!("Failed to read backends directory: {}", e))?;

    let mut candidates: Vec<(u32, String, PathBuf)> = Vec::new();
    for version_entry in version_dirs {
        let version_entry =
            version_entry.map_err(|e| format!("Failed to read version entry: {}", e))?;

        let version_path = version_entry.path();
        let version_name = match version_path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => continue,
        };

        if version_name == latest_version {
            continue;
        }

        let backend_type_path = version_path.join(&backend_type);
        if !backend_type_path.exists() || !is_backend_installed(&backend_type_path) {
            continue;
        }

        candidates.push((
            parse_backend_version(version_name.clone()),
            version_name,
            backend_type_path,
        ));
    }

    // Newest first, name as a tiebreaker so unparseable versions (all 0) are
    // still pruned deterministically.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

    if candidates.len() > keep_previous {
        let retained: Vec<&str> = candidates
            .iter()
            .take(keep_previous)
            .map(|(_, name, _)| name.as_str())
            .collect();
        log::info!(
            "Pruning {} of {}: keeping active {} plus {:?}",
            backend_type,
            latest_version,
            latest_version,
            retained
        );
    }

    for (_, version_name, path) in candidates.into_iter().skip(keep_previous) {
        match fs::remove_dir_all(&path) {
            Ok(_) => {
                log::info!(
                    "Removed old version {} of {}: {}",
                    version_name,
                    backend_type,
                    path.display()
                );
                removed_paths.push(path.to_string_lossy().to_string());
            }
            Err(e) => {
                log::warn!(
                    "Failed to remove old backend version: {} - {}",
                    path.display(),
                    e
                );
            }
        }
    }

    Ok(removed_paths)
}

#[tauri::command]
pub fn validate_backend_string(backend_string: String) -> Result<(String, String), String> {
    let parts: Vec<&str> = backend_string.split('/').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid backend format: {}", backend_string));
    }

    let version = parts[0].trim();
    let backend = parts[1].trim();

    if version.is_empty() || backend.is_empty() {
        return Err(format!("Invalid backend format: {}", backend_string));
    }

    Ok((version.to_string(), backend.to_string()))
}

#[tauri::command]
pub fn should_migrate_backend(
    stored_backend_type: String,
    version_backends: Vec<BackendInfo>,
) -> Result<Option<String>, String> {
    let mapped_new_backend_type = map_old_backend_to_new(stored_backend_type.clone());
    let is_migration_needed = mapped_new_backend_type != stored_backend_type;

    if !is_migration_needed {
        return Ok(None);
    }

    // Check if the new, mapped backend is available
    let is_new_type_available = version_backends
        .iter()
        .any(|vb| map_old_backend_to_new(vb.backend.clone()) == mapped_new_backend_type);

    if is_new_type_available {
        log::info!(
            "Migration needed from '{}' to '{}'",
            stored_backend_type,
            mapped_new_backend_type
        );
        Ok(Some(mapped_new_backend_type))
    } else {
        log::warn!(
            "Migration from '{}' to '{}' skipped: New type not available",
            stored_backend_type,
            mapped_new_backend_type
        );
        Ok(None)
    }
}

// ============================================================================
// Backend Path & Installation Commands
// ============================================================================

#[tauri::command]
pub fn get_backend_dir(backend: String, version: String, jan_data_folder: String) -> String {
    PathBuf::from(&jan_data_folder)
        .join("llamacpp")
        .join("backends")
        .join(&version)
        .join(&backend)
        .to_string_lossy()
        .to_string()
}

#[tauri::command]
pub fn get_backend_exe_path(
    backend: String,
    version: String,
    jan_data_folder: String,
    is_windows: bool,
) -> String {
    let backend_dir = PathBuf::from(get_backend_dir(backend, version, jan_data_folder));
    let exe_name = if is_windows {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let build_path = backend_dir.join("build").join("bin").join(exe_name);
    if build_path.exists() {
        return build_path.to_string_lossy().to_string();
    }

    backend_dir.join(exe_name).to_string_lossy().to_string()
}

#[tauri::command]
pub fn check_backend_installed(
    backend: String,
    version: String,
    jan_data_folder: String,
    is_windows: bool,
) -> bool {
    let exe_path =
        PathBuf::from(get_backend_exe_path(backend, version, jan_data_folder, is_windows));
    exe_path.exists()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BackendVerificationResult {
    pub verified: bool,
    pub missing_libraries: Vec<String>,
    pub resolved_libraries: Vec<String>,
}

fn gpu_backend_keyword(backend: &str) -> Option<&'static str> {
    let b = backend.to_lowercase();
    if b.contains("cuda") {
        Some("cuda")
    } else if b.contains("vulkan") {
        Some("vulkan")
    } else if b.contains("hip") || b.contains("rocm") {
        // Seeds the dep scan with libggml-hip.*, which pulls in hipblas/rocblas.
        Some("hip")
    } else {
        None
    }
}

fn is_shared_lib_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    if cfg!(target_os = "windows") {
        lower.ends_with(".dll")
    } else {
        lower.ends_with(".so") || lower.contains(".so.")
    }
}

fn find_gpu_libs(bin_dir: &Path, keyword: &str) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(bin_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_lowercase(),
            None => continue,
        };
        if name.contains(keyword) && is_shared_lib_name(&name) {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn verify_backend_dependencies(
    bin_dir: &Path,
    exe_path: &Path,
    backend: &str,
) -> Result<BackendVerificationResult, crate::error::LlamacppError> {
    if !bin_dir.exists() {
        return Err(crate::error::LlamacppError::new(
            crate::error::ErrorCode::BinaryNotFound,
            "Backend directory not found".into(),
            Some(bin_dir.to_string_lossy().to_string()),
        ));
    }

    let start = std::time::Instant::now();
    let mut missing: std::collections::HashSet<String> = std::collections::HashSet::new();

    let paths: Vec<PathBuf> = if cfg!(target_os = "macos") {
        Vec::new()
    } else {
        match gpu_backend_keyword(backend) {
            Some(kw) => find_gpu_libs(bin_dir, kw),
            None => Vec::new(),
        }
    };

    let _ = exe_path;

    // HIP/ROCm runtime libs (rocblas/hipblas/amdhip) live outside the backend
    // dir in versioned roots like /opt/rocm-7.2.0/lib; add them so the scan
    // doesn't false-flag them as missing.
    let mut lib_dirs: Vec<PathBuf> = vec![bin_dir.to_path_buf()];
    if gpu_backend_keyword(backend) == Some("hip") {
        for p in jan_utils::find_rocm_paths().lib_paths {
            lib_dirs.push(PathBuf::from(p));
        }
    }

    let analysis = crate::deps_analyzer::analyze_out_of_process(&lib_dirs, &paths);

    let resolved: std::collections::HashSet<String> = analysis.resolved.into_iter().collect();
    for name in analysis.missing {
        missing.insert(name);
    }

    // A lib may be resolved by one binary but missing for another — keep only
    // libs that no binary resolved.
    let mut truly_missing: Vec<String> =
        missing.difference(&resolved).cloned().collect();
    let mut resolved_vec: Vec<String> = resolved.into_iter().collect();
    truly_missing.sort();
    resolved_vec.sort();

    let elapsed = start.elapsed();
    log::info!(
        "verify_backend_dependencies: scanned {} in {:.2}s ({} resolved, {} missing)",
        bin_dir.display(),
        elapsed.as_secs_f64(),
        resolved_vec.len(),
        truly_missing.len(),
    );

    Ok(BackendVerificationResult {
        verified: truly_missing.is_empty(),
        missing_libraries: truly_missing,
        resolved_libraries: resolved_vec,
    })
}

#[tauri::command]
pub async fn verify_backend_installation(
    backend: String,
    version: String,
    jan_data_folder: String,
    is_windows: bool,
) -> Result<BackendVerificationResult, crate::error::LlamacppError> {
    let exe_path =
        PathBuf::from(get_backend_exe_path(backend.clone(), version.clone(), jan_data_folder.clone(), is_windows));
    if !exe_path.exists() {
        return Err(crate::error::LlamacppError::new(
            crate::error::ErrorCode::BinaryNotFound,
            "Backend executable not found".into(),
            Some(exe_path.to_string_lossy().to_string()),
        ));
    }

    #[cfg(target_os = "linux")]
    if jan_utils::system::is_flatpak() {
        return Ok(BackendVerificationResult {
            verified: true,
            missing_libraries: Vec::new(),
            resolved_libraries: Vec::new(),
        });
    }
    // Libs sit next to the exe (build/bin/ when present); backend_dir is too high.
    let bin_dir = exe_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(get_backend_dir(backend.clone(), version, jan_data_folder)));
    tokio::task::spawn_blocking(move || verify_backend_dependencies(&bin_dir, &exe_path, &backend))
        .await
        .map_err(|e| crate::error::LlamacppError::new(
            crate::error::ErrorCode::InternalError,
            "Dependency verification task panicked".into(),
            Some(e.to_string()),
        ))?
}

// ============================================================================
// Remote Backend Fetching
// ============================================================================

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
}

/// Subset of the app's ProxyConfig needed to build a reqwest proxy. Kept local
/// to avoid a cross-crate dependency on the app crate. Shape must stay in sync
/// with the TS `getProxyConfig()` payload and the app's `ProxyConfig` struct.
#[derive(Deserialize, Clone, Debug)]
pub struct ProxyConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub ignore_ssl: Option<bool>,
}

fn build_http_client(proxy: Option<&ProxyConfig>) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().user_agent("jan-app");

    if let Some(cfg) = proxy {
        if !cfg.url.trim().is_empty() {
            let mut p = reqwest::Proxy::all(&cfg.url)
                .map_err(|e| format!("Invalid proxy URL: {}", e))?;
            if let (Some(u), Some(pw)) = (&cfg.username, &cfg.password) {
                if !u.is_empty() {
                    p = p.basic_auth(u, pw);
                }
            }
            builder = builder.proxy(p);
            if cfg.ignore_ssl.unwrap_or(false) {
                builder = builder.danger_accept_invalid_certs(true);
            }
        }
    }

    builder
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

#[tauri::command]
pub async fn fetch_remote_supported_backends(
    supported_backends: Vec<String>,
    proxy: Option<ProxyConfig>,
) -> Result<Vec<BackendInfo>, String> {
    let client = build_http_client(proxy.as_ref())?;

    let resp = client
        .get(RELEASES_API)
        .send()
        .await
        .map_err(|e| format!("Failed to reach the llama.cpp releases API: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "llama.cpp releases API returned HTTP {}",
            resp.status()
        ));
    }
    let releases: Vec<GithubRelease> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitHub releases JSON: {}", e))?;

    // Build numbers are `b<n>`, so a lexical sort puts b982 above b9100.
    let mut sorted_releases = releases;
    sorted_releases.sort_by_key(|r| std::cmp::Reverse(parse_backend_version(r.tag_name.clone())));

    let mut result: Vec<BackendInfo> = Vec::new();
    for release in sorted_releases.into_iter().take(10) {
        let version = &release.tag_name;
        let prefix = format!("llama-{}-bin-", version);
        for asset in &release.assets {
            let Some(rest) = asset.name.strip_prefix(&prefix) else {
                continue;
            };
            let Some(token) = rest
                .strip_suffix(".zip")
                .or_else(|| rest.strip_suffix(".tar.gz"))
            else {
                continue;
            };
            let Some(parsed) = parse_backend_token(token) else {
                continue;
            };
            let key = capability_key(&parsed.os, &parsed.arch, &parsed.family, parsed.cuda_major);
            if supported_backends.contains(&key) {
                result.push(BackendInfo {
                    version: version.clone(),
                    backend: token.to_string(),
                });
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Backend Download Item Builder
// ============================================================================

#[derive(Serialize, Deserialize)]
pub struct BackendDownloadItem {
    pub url: String,
    pub save_path: String,
    pub model_id: String,
}

/// Internal helper: check if a CUDA runtime library is present at the new
/// location (`backend_dir/build/bin/{libname}`) without performing any
/// migration side-effects.
/// Whether the cudart redistributable is already unpacked next to the server
/// binary. Upstream archives are flat, so the DLLs land in `build/bin/`
/// alongside `llama-server.exe` once the layout is normalized.
fn check_cuda_installed_internal(backend_dir: &str, cuda_major: u32, os_type: &str) -> bool {
    let libname: &str = match (os_type, cuda_major) {
        ("windows", 12) => "cudart64_12.dll",
        ("windows", 13) => "cudart64_13.dll",
        ("linux", 12) => "libcudart.so.12",
        ("linux", 13) => "libcudart.so.13",
        _ => return false,
    };

    PathBuf::from(backend_dir)
        .join("build")
        .join("bin")
        .join(libname)
        .exists()
}

#[tauri::command]
pub fn build_backend_download_items(
    backend: String,
    version: String,
    source: String,
    jan_data_folder: String,
    os_type: String,
) -> Result<Vec<BackendDownloadItem>, String> {
    let _ = source;
    let token = parse_backend_token(&backend)
        .ok_or_else(|| format!("Unrecognised backend id: {}", backend))?;

    let backend_dir = get_backend_dir(backend.clone(), version.clone(), jan_data_folder);
    let task_id = format!("llamacpp-{}-{}", version, backend).replace('.', "-");

    // Upstream ships Windows as .zip and everything else as .tar.gz.
    let archive_ext = if token.os == "win" { "zip" } else { "tar.gz" };

    let mut items = vec![BackendDownloadItem {
        url: format!(
            "{}/{}/llama-{}-bin-{}.{}",
            RELEASE_DOWNLOAD, version, version, backend, archive_ext
        ),
        save_path: format!("{}/backend.{}", backend_dir, archive_ext),
        model_id: task_id.clone(),
    }];

    // CUDA builds link against the redistributable dynamically, published as a
    // sibling asset named for the same token.
    if let Some(major) = token.cuda_major {
        if !check_cuda_installed_internal(&backend_dir, major, &os_type) {
            items.push(BackendDownloadItem {
                url: format!(
                    "{}/{}/cudart-llama-bin-{}.zip",
                    RELEASE_DOWNLOAD, version, backend
                ),
                save_path: format!("{}/build/bin/cudart.zip", backend_dir),
                model_id: task_id.clone(),
            });
        }
    }

    Ok(items)
}

// ============================================================================
// Settings Update Handler
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SettingUpdateResult {
    pub backend_type_updated: bool,
    pub effective_backend_type: Option<String>,
    pub needs_backend_installation: bool,
    pub version: Option<String>,
    pub backend: Option<String>,
}

#[tauri::command]
pub fn handle_setting_update(
    key: String,
    value: String,
    current_stored_backend: Option<String>,
) -> Result<SettingUpdateResult, String> {
    if key != "version_backend" {
        // For non-backend settings, return a simple result
        return Ok(SettingUpdateResult {
            backend_type_updated: false,
            effective_backend_type: None,
            needs_backend_installation: false,
            version: None,
            backend: None,
        });
    }

    // Handle version_backend update
    let parts: Vec<&str> = value.split('/').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid backend format: {}", value));
    }

    let version = parts[0].to_string();
    let backend = parts[1].to_string();

    if version.is_empty() || backend.is_empty() {
        return Err(format!("Invalid backend format: {}", value));
    }

    // Get the effective/migrated backend type
    let effective_backend_type = map_old_backend_to_new(backend.clone());

    // Check if backend type changed
    let backend_type_updated = match current_stored_backend {
        Some(stored) => stored != effective_backend_type,
        None => true,
    };

    log::info!(
        "Setting update for version_backend: {}/{} (effective: {})",
        version,
        backend,
        effective_backend_type
    );

    Ok(SettingUpdateResult {
        backend_type_updated,
        effective_backend_type: Some(effective_backend_type),
        needs_backend_installation: true,
        version: Some(version),
        backend: Some(backend),
    })
}

// ---------------------------- Tests ------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn install_backend(root: &Path, version: &str, backend: &str) {
        let dir = root.join(version).join(backend);
        std::fs::create_dir_all(&dir).unwrap();
        let exe = if cfg!(target_os = "windows") {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        File::create(dir.join(exe)).unwrap();
    }

    fn versions_left(root: &Path, backend: &str) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join(backend).exists())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        v.sort();
        v
    }

    #[tokio::test]
    async fn retention_keeps_the_active_version_and_n_previous() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for v in ["b900", "b1000", "b1100", "b1200"] {
            install_backend(root, v, "cpu");
        }

        let removed = remove_old_backend_versions(
            root.to_string_lossy().to_string(),
            "b1200".into(),
            "cpu".into(),
            2,
        )
        .await
        .unwrap();

        assert_eq!(removed.len(), 1);
        assert_eq!(versions_left(root, "cpu"), vec!["b1000", "b1100", "b1200"]);
    }

    #[tokio::test]
    async fn retention_orders_by_build_number_not_lexically() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Lexically "b982" > "b1100"; by build number it is older and must go.
        for v in ["b982", "b1100", "b1200"] {
            install_backend(root, v, "cpu");
        }

        remove_old_backend_versions(
            root.to_string_lossy().to_string(),
            "b1200".into(),
            "cpu".into(),
            1,
        )
        .await
        .unwrap();

        assert_eq!(versions_left(root, "cpu"), vec!["b1100", "b1200"]);
    }

    #[tokio::test]
    async fn retention_never_removes_the_active_version() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        install_backend(root, "b1000", "cpu");

        let removed = remove_old_backend_versions(
            root.to_string_lossy().to_string(),
            "b1000".into(),
            "cpu".into(),
            0,
        )
        .await
        .unwrap();

        assert!(removed.is_empty());
        assert_eq!(versions_left(root, "cpu"), vec!["b1000"]);
    }

    #[tokio::test]
    async fn retention_ignores_other_backend_types() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        install_backend(root, "b900", "cpu");
        install_backend(root, "b900", "cuda");
        install_backend(root, "b1000", "cpu");

        remove_old_backend_versions(
            root.to_string_lossy().to_string(),
            "b1000".into(),
            "cpu".into(),
            0,
        )
        .await
        .unwrap();

        // The cuda install under b900 survives; only its cpu sibling is pruned.
        assert!(root.join("b900").join("cuda").exists());
        assert!(!root.join("b900").join("cpu").exists());
    }

    #[test]
    fn digest_normalization_tolerates_producer_quirks() {
        // Windows CertUtil output: uppercase, BOM, folded whitespace.
        assert_eq!(normalize_digest("\u{feff}AB\n  cd \r\n"), "abcd");
        assert_eq!(normalize_digest("  abcd  "), "abcd");
        assert_eq!(normalize_digest(""), "");
        // Non-hex noise cannot smuggle itself into a digest.
        assert_eq!(normalize_digest("zz12zz"), "12");
    }

    #[test]
    fn checksum_manifest_parses_the_published_shape() {
        let yaml = "version: b1234\nfiles:\n\
                    - url: llama-b1234-bin-linux-x64.tar.gz\n  sha512: >-\n    ABCD\n  size: 10\n\
                    - url: cudart-llama-bin-linux-cu12.0-x64.tar.gz\n  sha512: >-\n    beef\n  size: 20\n";
        let m: ChecksumManifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(m.files.len(), 2);
        assert_eq!(m.files[0].url, "llama-b1234-bin-linux-x64.tar.gz");
        assert_eq!(normalize_digest(&m.files[0].sha512), "abcd");
    }

    #[tokio::test]
    async fn sha512_verification_accepts_and_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.tar.gz");
        std::fs::write(&path, b"jan").unwrap();

        let actual = {
            use sha2::{Digest, Sha512};
            let mut h = Sha512::new();
            h.update(b"jan");
            format!("{:x}", h.finalize())
        };
        let p = path.to_string_lossy().to_string();

        assert!(verify_file_sha512(p.clone(), actual.to_uppercase())
            .await
            .unwrap());
        assert!(!verify_file_sha512(p.clone(), "dead".repeat(32))
            .await
            .unwrap());
        // No published digest means nothing to contradict.
        assert!(verify_file_sha512(p.clone(), String::new()).await.unwrap());
        // A malformed (too-short) digest is fail-soft, not a mismatch.
        assert!(verify_file_sha512(p, "zz12zz".into()).await.unwrap());
    }

    #[test]
    fn test_get_backend_category_hip() {
        assert_eq!(
            get_backend_category("win-rocm-10.0-x64").as_deref(),
            Some("rocm")
        );
        assert_eq!(
            get_backend_category("ubuntu-rocm-10.0-x64").as_deref(),
            Some("rocm")
        );
    }

    // --- Tests for compare_versions (Private helper) ---

    #[test]
    fn test_compare_versions() {
        assert_eq!(compare_versions("1.0", "2.0"), -1);
        assert_eq!(compare_versions("2.0", "1.0"), 1);
        assert_eq!(compare_versions("1.0", "1.0"), 0);
        assert_eq!(compare_versions("1.0.1", "1.0"), 1);
        assert_eq!(compare_versions("450.80.02", "450.80.02"), 0);
        assert_eq!(compare_versions("525.60.13", "450.80.02"), 1);
        assert_eq!(compare_versions("10", "2"), 1); // Numeric check, not string
    }

    // --- Tests for get_supported_features ---

    #[test]
    fn test_get_supported_features_cpu_only() {
        let gpus = vec![];
        let exts = vec!["avx".to_string(), "avx2".to_string()];

        let result = get_supported_features("linux".to_string(), exts, gpus).unwrap();

        assert!(result.avx);
        assert!(result.avx2);
        assert!(!result.avx512);
        assert!(!result.vulkan);
    }

    #[test]
    fn test_get_supported_features_cuda_linux() {
        // Driver 525.60.13 supports CUDA 12 on Linux
        let gpus = vec![GpuInfo {
            driver_version: "530.00".to_string(),
            vendor: Some("NVIDIA".to_string()),
            nvidia_info: Some(NvidiaInfo {
                compute_capability: "8.0".to_string(),
            }),
            vulkan_info: None,
        }];

        let result = get_supported_features("linux".to_string(), vec![], gpus).unwrap();

        assert!(result.cuda12); // 530 > 525
        assert!(!result.cuda13); // 530 < 580
        assert!(!result.hip); // NVIDIA GPU, no HIP
    }

    #[test]
    fn test_get_supported_features_vulkan() {
        let gpus = vec![GpuInfo {
            driver_version: "0.0".to_string(),
            vendor: Some("Intel".to_string()),
            nvidia_info: None,
            vulkan_info: Some(VulkanInfo {
                api_version: "1.3".to_string(),
            }),
        }];

        let result = get_supported_features("windows".to_string(), vec![], gpus).unwrap();

        assert!(result.vulkan);
        assert!(!result.hip); // Intel GPU, no HIP
    }

    #[test]
    fn test_get_supported_features_amd_hip() {
        // AMD GPU exposes Vulkan but no CUDA; HIP should be offered alongside.
        let gpus = vec![GpuInfo {
            driver_version: "0.0".to_string(),
            vendor: Some("AMD".to_string()),
            nvidia_info: None,
            vulkan_info: Some(VulkanInfo {
                api_version: "1.3".to_string(),
            }),
        }];

        let linux = get_supported_features("linux".to_string(), vec![], gpus).unwrap();
        assert!(linux.hip);
        assert!(linux.vulkan);

        // macOS never offers HIP even with an AMD GPU.
        let mac_gpus = vec![GpuInfo {
            driver_version: "0.0".to_string(),
            vendor: Some("AMD".to_string()),
            nvidia_info: None,
            vulkan_info: None,
        }];
        let mac = get_supported_features("macos".to_string(), vec![], mac_gpus).unwrap();
        assert!(!mac.hip);
    }

    // --- Tests for determine_supported_backends ---

    #[test]
    fn test_determine_supported_backends_windows_all() {
        let features = SystemFeatures {
            cuda12: true,
            cuda13: false,
            vulkan: true,
            hip: true,
        };

        let result =
            determine_supported_backends("windows".to_string(), "x86_64".to_string(), features)
                .unwrap();

        // Capability keys, not asset names: upstream moves the CUDA minor and
        // the ROCm version between releases, so only the family is pinned here.
        assert!(result.contains(&"win-x64/cpu".to_string()));
        assert!(result.contains(&"win-x64/cuda12".to_string()));
        // The driver is too old for CUDA 13, so that build is not offered.
        assert!(!result.contains(&"win-x64/cuda13".to_string()));
        assert!(result.contains(&"win-x64/vulkan".to_string()));
        assert!(result.contains(&"win-x64/rocm".to_string()));
    }

    #[test]
    fn test_determine_supported_backends_mac_arm() {
        let features = SystemFeatures {
            cuda12: false,
            cuda13: false,
            vulkan: false,
            hip: false,
        };

        let result =
            determine_supported_backends("macos".to_string(), "arm64".to_string(), features)
                .unwrap();

        // macOS ships Metal inside the CPU archive; there is no separate GPU
        // build to offer.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "macos-arm64/cpu");
    }

    // --- Tests for list_supported_backends ---

    #[tokio::test]
    async fn test_list_supported_backends_sorting_and_dedup() {
        let remote = vec![
            BackendInfo {
                version: "b7523".into(),
                backend: "backend-a".into(),
            },
            BackendInfo {
                version: "b7523".into(),
                backend: "backend-b".into(),
            },
        ];

        let local = vec![
            // Should override remote
            BackendInfo {
                version: "b7523".into(),
                backend: "backend-a".into(),
            },
            // Newer version
            BackendInfo {
                version: "b7524".into(),
                backend: "backend-c".into(),
            },
        ];

        let result = list_supported_backends(remote, local).await.unwrap();

        // Expect 3 items: b7524(c), b7523(a), b7523(b)
        assert_eq!(result.len(), 3);

        // Check sorting: Version desc (b7524 > b7523)
        assert_eq!(result[0].version, "b7524");
        assert_eq!(result[1].version, "b7523");
        assert_eq!(result[2].version, "b7523");

        // Check sorting: Backend asc for same version
        // backend-a comes before backend-b
        assert_eq!(result[1].backend, "backend-a");
        assert_eq!(result[2].backend, "backend-b");
    }

    // --- Tests for parse_backend_version ---
    #[test]
    fn test_parse_backend_version() {
        assert_eq!(parse_backend_version("b7523".to_string()), 7523);
        assert_eq!(parse_backend_version("b7524".to_string()), 7524);
        assert_eq!(parse_backend_version("7525".to_string()), 7525);
        assert_eq!(parse_backend_version("v100".to_string()), 100);
        assert_eq!(parse_backend_version("invalid".to_string()), 0);
        // Note: "v1.0.0" would fail to parse as u32 due to dots, returning 0
        assert_eq!(parse_backend_version("v1.0.0".to_string()), 0);
    }
    // --- Filesystem Integration Tests ---

    #[tokio::test]
    async fn test_get_local_installed_backends() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path();

        // Structure:
        // root/
        //   b7523/
        //     backend-a/
        //       build/bin/llama-server (exe)
        //     backend-empty/ (no exe)

        let v1_path = root.join("b7523");
        let backend_a = v1_path.join("backend-a");
        let backend_empty = v1_path.join("backend-empty");

        fs::create_dir_all(backend_a.join("build").join("bin")).unwrap();
        fs::create_dir_all(&backend_empty).unwrap();

        // Create mock executable
        let exe_name = if cfg!(target_os = "windows") {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        File::create(backend_a.join("build").join("bin").join(exe_name)).unwrap();

        let result = get_local_installed_backends(root.to_string_lossy().to_string())
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].version, "b7523");
        assert_eq!(result[0].backend, "backend-a");
    }

    #[tokio::test]
    async fn test_is_cuda_installed_migration() {
        let backend_dir = tempfile::tempdir().unwrap();
        let jan_data_dir = tempfile::tempdir().unwrap();

        let version = "12.0";
        let os_type = "linux"; // Maps to libcudart.so.12

        // Setup Old Path: jan_data/llamacpp/lib/libcudart.so.12
        let old_lib_dir = jan_data_dir.path().join("llamacpp").join("lib");
        fs::create_dir_all(&old_lib_dir).unwrap();
        let lib_name = "libcudart.so.12";
        let old_file_path = old_lib_dir.join(lib_name);
        {
            let mut f = File::create(&old_file_path).unwrap();
            f.write_all(b"dummy content").unwrap();
        }

        // Run Check (should trigger migration)
        let installed = is_cuda_installed(
            backend_dir.path().to_string_lossy().to_string(),
            version.to_string(),
            os_type.to_string(),
            jan_data_dir.path().to_string_lossy().to_string(),
        )
        .await
        .unwrap();

        assert!(installed, "Should return true after migration");

        // Verify Migration
        let new_path = backend_dir.path().join("build").join("bin").join(lib_name);
        assert!(new_path.exists(), "File should exist in new location");
        assert!(
            !old_file_path.exists(),
            "File should be removed from old location"
        );
    }

    #[tokio::test]
    async fn test_is_cuda_installed_already_exists() {
        let backend_dir = tempfile::tempdir().unwrap();
        let jan_data_dir = tempfile::tempdir().unwrap(); // Empty

        let version = "11.7";
        let os_type = "windows"; // Maps to cudart64_110.dll
        let lib_name = "cudart64_110.dll";

        // Setup New Path directly
        let target_dir = backend_dir.path().join("build").join("bin");
        fs::create_dir_all(&target_dir).unwrap();
        File::create(target_dir.join(lib_name)).unwrap();

        let installed = is_cuda_installed(
            backend_dir.path().to_string_lossy().to_string(),
            version.to_string(),
            os_type.to_string(),
            jan_data_dir.path().to_string_lossy().to_string(),
        )
        .await
        .unwrap();

        assert!(installed);
    }

    // --- Tests for find_latest_version_for_backend ---

    #[test]
    fn test_find_latest_version_for_backend() {
        let backends = vec![
            BackendInfo {
                version: "b7523".into(),
                backend: "linux-common_cpus-x64".into(),
            },
            BackendInfo {
                version: "b7524".into(),
                backend: "linux-common_cpus-x64".into(),
            },
            BackendInfo {
                version: "b7522".into(),
                backend: "linux-common_cpus-x64".into(),
            },
        ];

        let result = find_latest_version_for_backend(backends, "linux-common_cpus-x64".to_string());
        assert_eq!(result, Some("b7524/linux-common_cpus-x64".to_string()));
    }

    #[test]
    fn test_find_latest_version_for_backend_with_migration() {
        let backends = vec![
            BackendInfo {
                version: "b7523".into(),
                backend: "linux-avx2-x64".into(), // Old format
            },
            BackendInfo {
                version: "b7524".into(),
                backend: "linux-common_cpus-x64".into(), // New format
            },
        ];

        // Both should map to linux-common_cpus-x64
        let result = find_latest_version_for_backend(backends, "linux-common_cpus-x64".to_string());
        assert_eq!(result, Some("b7524/linux-common_cpus-x64".to_string()));
    }

    // --- Tests for check_backend_for_updates ---

    #[tokio::test]
    async fn test_check_backend_for_updates_needs_update() {
        let current = "b7523/linux-common_cpus-x64".to_string();
        let available = vec![
            BackendInfo {
                version: "b7523".into(),
                backend: "linux-common_cpus-x64".into(),
            },
            BackendInfo {
                version: "b7524".into(),
                backend: "linux-common_cpus-x64".into(),
            },
        ];

        let result = check_backend_for_updates(current, available).await.unwrap();

        assert!(result.update_needed);
        assert_eq!(result.new_version, "b7524");
        assert_eq!(
            result.target_backend,
            Some("b7524/linux-common_cpus-x64".to_string())
        );
    }

    #[tokio::test]
    async fn test_check_backend_for_updates_already_latest() {
        let current = "b7524/linux-common_cpus-x64".to_string();
        let available = vec![
            BackendInfo {
                version: "b7523".into(),
                backend: "linux-common_cpus-x64".into(),
            },
            BackendInfo {
                version: "b7524".into(),
                backend: "linux-common_cpus-x64".into(),
            },
        ];

        let result = check_backend_for_updates(current, available).await.unwrap();

        assert!(!result.update_needed);
        assert_eq!(result.new_version, "0");
        assert_eq!(result.target_backend, None);
    }

    // --- Tests for validate_backend_string ---

    #[test]
    fn test_validate_backend_string_valid() {
        let result = validate_backend_string("b7524/linux-common_cpus-x64".to_string()).unwrap();
        assert_eq!(result.0, "b7524");
        assert_eq!(result.1, "linux-common_cpus-x64");
    }

    #[test]
    fn test_validate_backend_string_invalid() {
        let result = validate_backend_string("invalid-format".to_string());
        assert!(result.is_err());
    }

    // --- Tests for should_migrate_backend ---

    #[test]
    fn test_should_migrate_backend_no_migration_needed() {
        let new_backend = "linux-common_cpus-x64".to_string();
        let available = vec![BackendInfo {
            version: "b7524".into(),
            backend: "linux-common_cpus-x64".into(),
        }];

        let result = should_migrate_backend(new_backend, available).unwrap();
        assert_eq!(result, None);
    }

    // -------------------------------------------------------------------------
    // Tests for get_backend_dir
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_backend_dir_path_format() {
        let result = get_backend_dir(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            "/home/user/.jan".to_string(),
        );
        assert!(result.contains("llamacpp"));
        assert!(result.contains("backends"));
        assert!(result.contains("b7523"));
        assert!(result.contains("linux-common_cpus-x64"));
        // Ensure the segments are in the right order
        let path = PathBuf::from(&result);
        let components: Vec<_> = path.components().collect();
        let names: Vec<String> = components
            .iter()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        let llamacpp_idx = names.iter().position(|n| n == "llamacpp").unwrap();
        let backends_idx = names.iter().position(|n| n == "backends").unwrap();
        let version_idx = names.iter().position(|n| n == "b7523").unwrap();
        let backend_idx = names.iter().position(|n| n == "linux-common_cpus-x64").unwrap();
        assert!(llamacpp_idx < backends_idx);
        assert!(backends_idx < version_idx);
        assert!(version_idx < backend_idx);
    }

    // -------------------------------------------------------------------------
    // Tests for get_backend_exe_path
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_backend_exe_path_no_build_dir_linux() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let result = get_backend_exe_path(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data.clone(),
            false,
        );
        // Should fall back to root dir (build dir doesn't exist)
        assert!(result.ends_with("llama-server"));
        assert!(!result.ends_with("llama-server.exe"));
    }

    #[test]
    fn test_get_backend_exe_path_no_build_dir_windows() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let result = get_backend_exe_path(
            "win-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            true,
        );
        assert!(result.ends_with("llama-server.exe"));
    }

    #[test]
    fn test_get_backend_exe_path_with_build_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        // Create the build/bin directory and place the exe there
        let backend_dir = PathBuf::from(&jan_data)
            .join("llamacpp")
            .join("backends")
            .join("b7523")
            .join("linux-common_cpus-x64");
        let build_bin = backend_dir.join("build").join("bin");
        fs::create_dir_all(&build_bin).unwrap();
        File::create(build_bin.join("llama-server")).unwrap();

        let result = get_backend_exe_path(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            false,
        );
        assert!(result.contains("build"));
        assert!(result.contains("bin"));
        assert!(result.ends_with("llama-server"));
    }

    // -------------------------------------------------------------------------
    // Tests for check_backend_installed
    // -------------------------------------------------------------------------

    #[test]
    fn test_check_backend_installed_true() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let backend_dir = PathBuf::from(&jan_data)
            .join("llamacpp")
            .join("backends")
            .join("b7523")
            .join("linux-common_cpus-x64");
        fs::create_dir_all(&backend_dir).unwrap();
        File::create(backend_dir.join("llama-server")).unwrap();

        let result = check_backend_installed(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            false,
        );
        assert!(result);
    }

    #[test]
    fn test_check_backend_installed_false_when_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let result = check_backend_installed(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            false,
        );
        assert!(!result);
    }

    // -------------------------------------------------------------------------
    // Tests for verify_backend_installation
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_verify_backend_installation_returns_err_when_exe_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let result = verify_backend_installation(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            false,
        )
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.code, crate::error::ErrorCode::BinaryNotFound));
    }

    #[test]
    fn test_verify_backend_dependencies_returns_err_on_missing_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let nonexistent = temp_dir.path().join("does-not-exist");

        let exe = nonexistent.join("llama-server");
        let result = verify_backend_dependencies(&nonexistent, &exe, "linux-common_cpus-x64");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.code, crate::error::ErrorCode::BinaryNotFound));
    }

    #[test]
    fn test_verify_backend_dependencies_skips_unparseable_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        // An unparseable file in the dir should be silently skipped, not error.
        fs::write(temp_dir.path().join("llama-server"), b"not a real binary").unwrap();
        fs::write(temp_dir.path().join("libggml.so"), b"not a real binary").unwrap();

        let dir = temp_dir.path().to_path_buf();
        let exe = dir.join("llama-server");
        let result = verify_backend_dependencies(&dir, &exe, "linux-common_cpus-x64");

        // Succeeds with an empty result — nothing parseable, nothing missing.
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.verified);
        assert!(r.missing_libraries.is_empty());
    }

    #[tokio::test]
    async fn test_verify_backend_installation_scans_whole_backend_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        // Create the backend dir with a stub exe and a stub DLL beside it.
        let backend_dir = PathBuf::from(&jan_data)
            .join("llamacpp")
            .join("backends")
            .join("b7523")
            .join("linux-common_cpus-x64");
        fs::create_dir_all(&backend_dir).unwrap();
        fs::write(backend_dir.join("llama-server"), b"not a real binary").unwrap();
        fs::write(backend_dir.join("libggml.so"), b"not a real binary").unwrap();

        // Both files are unparseable stubs — verify_backend_installation must
        // return Ok (nothing missing) rather than Err, confirming it walks the
        // whole directory and skips non-binaries rather than hard-failing.
        let result = verify_backend_installation(
            "linux-common_cpus-x64".to_string(),
            "b7523".to_string(),
            jan_data,
            false,
        )
        .await;
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.verified);
    }

    #[test]
    fn test_is_virtual_windows_dll() {
        assert!(crate::deps_analyzer::is_virtual_windows_dll("api-ms-win-core-memory-l1-1-0.dll"));
        assert!(crate::deps_analyzer::is_virtual_windows_dll("API-MS-WIN-CORE-LIBRARYLOADER-L1-2-0.DLL"));
        assert!(crate::deps_analyzer::is_virtual_windows_dll("ext-ms-win-ntuser-draw-l1-1-0.dll"));
        assert!(!crate::deps_analyzer::is_virtual_windows_dll("KERNEL32.dll"));
        assert!(!crate::deps_analyzer::is_virtual_windows_dll("libcuda.so.1"));
        assert!(!crate::deps_analyzer::is_virtual_windows_dll("libvulkan.so.1"));
    }

    // -------------------------------------------------------------------------
    // Tests for build_backend_download_items
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_backend_download_items_non_cuda_returns_one_item() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let items = build_backend_download_items(
            "ubuntu-x64".to_string(),
            "b10883".to_string(),
            "github".to_string(),
            jan_data,
            "linux".to_string(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        assert!(items[0].url.contains("ggml-org/llama.cpp"));
        assert!(items[0].url.ends_with("llama-b10883-bin-ubuntu-x64.tar.gz"));
    }

    #[test]
    fn test_build_backend_download_items_cuda_adds_the_matching_cudart() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let items = build_backend_download_items(
            "win-cuda-12.4-x64".to_string(),
            "b10883".to_string(),
            "github".to_string(),
            jan_data,
            "windows".to_string(),
        )
        .unwrap();

        assert_eq!(items.len(), 2);
        assert!(items[0]
            .url
            .ends_with("/b10883/llama-b10883-bin-win-cuda-12.4-x64.zip"));
        // The redistributable is named for the same token as the backend, so
        // a CUDA minor bump upstream needs no change here.
        assert!(items[1]
            .url
            .ends_with("/b10883/cudart-llama-bin-win-cuda-12.4-x64.zip"));
        // All items must share the same model_id for unified progress tracking
        assert_eq!(items[0].model_id, items[1].model_id);
    }

    #[test]
    fn test_build_backend_download_items_linux_uses_tar_gz() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        let items = build_backend_download_items(
            "ubuntu-vulkan-x64".to_string(),
            "b10883".to_string(),
            "github".to_string(),
            jan_data,
            "linux".to_string(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        assert!(items[0].url.ends_with(".tar.gz"));
        assert!(items[0].save_path.ends_with("backend.tar.gz"));
    }

    #[test]
    fn test_build_backend_download_items_rejects_an_unknown_token() {
        let temp_dir = tempfile::tempdir().unwrap();
        let jan_data = temp_dir.path().to_string_lossy().to_string();

        assert!(build_backend_download_items(
            "win-avx2-cuda-cu12.0-x64".to_string(),
            "b9967".to_string(),
            "github".to_string(),
            jan_data,
            "windows".to_string(),
        )
        .is_err());
    }

    // A release carries assets kuru has no business installing (sycl, openvino,
    // opencl, s390x). They must not reach the version list as bare tokens.
    #[test]
    fn test_parse_backend_token_families() {
        let cuda = parse_backend_token("win-cuda-12.4-x64").unwrap();
        assert_eq!(cuda.os, "win");
        assert_eq!(cuda.arch, "x64");
        assert_eq!(cuda.family, "cuda");
        assert_eq!(cuda.cuda_major, Some(12));

        // No family segment: `ubuntu-x64` and `macos-arm64` are CPU builds.
        assert_eq!(parse_backend_token("ubuntu-x64").unwrap().family, "cpu");
        assert_eq!(parse_backend_token("macos-arm64").unwrap().family, "cpu");
        assert_eq!(parse_backend_token("win-rocm-10.0-x64").unwrap().family, "rocm");

        assert!(parse_backend_token("ubuntu-sycl-fp16-x64").is_none());
        assert!(parse_backend_token("win-openvino-2026.3.1-x64").is_none());
        assert!(parse_backend_token("win-opencl-adreno-arm64").is_none());
    }

    // The whole point of keying on family rather than the full token: upstream
    // moves the CUDA minor between releases, and an update check that did not
    // group them would report "already latest" forever.
    #[test]
    fn test_backend_category_groups_across_a_cuda_minor_bump() {
        assert_eq!(
            get_backend_category("win-cuda-12.4-x64"),
            get_backend_category("win-cuda-12.6-x64")
        );
        assert_ne!(
            get_backend_category("win-cuda-12.4-x64"),
            get_backend_category("win-cuda-13.3-x64")
        );
        assert_eq!(get_backend_category("ubuntu-x64").as_deref(), Some("cpu"));
    }
}

// ============================================================================
// Release checksum verification
// ============================================================================

#[derive(Deserialize)]
struct ChecksumEntry {
    url: String,
    sha512: String,
}

#[derive(Deserialize)]
struct ChecksumManifest {
    files: Vec<ChecksumEntry>,
}

/// Normalize a digest as published. The Windows job produces it via `CertUtil`
/// piped through `Out-File`, so it can arrive uppercase and carrying a BOM or
/// UTF-16 padding; the YAML folded scalar (`>-`) can also fold in whitespace.
/// Keeping only hex digits is the one rule that holds for every producer.
fn normalize_digest(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_ascii_hexdigit())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Fetch `checksum.yml` for a release and return `filename -> sha512`.
///
/// Returns an empty map rather than an error when the manifest is absent or
/// unparseable: releases published before the manifest existed (and those
/// whose entries predate the `-bin-` naming fix) must stay installable.
#[tauri::command]
pub async fn fetch_backend_checksums(
    version: String,
    source: String,
    proxy: Option<ProxyConfig>,
) -> Result<HashMap<String, String>, String> {
    // ggml-org publishes no checksum manifest, so this 404s and verification is
    // skipped. Kept rather than deleted: the fetch is fail-soft by design, and
    // the day upstream starts publishing one this picks it up with no change.
    let _ = source;
    let url = format!("{}/{}/checksum.yml", RELEASE_DOWNLOAD, version);

    let client = build_http_client(proxy.as_ref())?;
    let resp = match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            log::warn!("checksum.yml for {} returned HTTP {}", version, r.status());
            return Ok(HashMap::new());
        }
        Err(e) => {
            log::warn!("Could not fetch checksum.yml for {}: {}", version, e);
            return Ok(HashMap::new());
        }
    };

    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => {
            log::warn!("Could not read checksum.yml for {}: {}", version, e);
            return Ok(HashMap::new());
        }
    };

    match serde_yaml::from_str::<ChecksumManifest>(&body) {
        Ok(m) => Ok(m
            .files
            .into_iter()
            .map(|f| (f.url, normalize_digest(&f.sha512)))
            .filter(|(_, digest)| !digest.is_empty())
            .collect()),
        Err(e) => {
            log::warn!("Could not parse checksum.yml for {}: {}", version, e);
            Ok(HashMap::new())
        }
    }
}

/// Streams the file so a multi-GB archive is never held in memory.
#[tauri::command]
pub async fn verify_file_sha512(path: String, expected: String) -> Result<bool, String> {
    let expected = normalize_digest(&expected);
    if expected.is_empty() {
        return Ok(true);
    }
    if expected.len() != 128 {
        log::warn!("Malformed SHA-512 digest (expected 128 hex chars, got {}), skipping verification", expected.len());
        return Ok(true);
    }

    tokio::task::spawn_blocking(move || {
        use sha2::{Digest, Sha512};
        use std::io::Read;

        let mut file = fs::File::open(&path).map_err(|e| format!("{}: {}", path, e))?;
        let mut hasher = Sha512::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf).map_err(|e| format!("{}: {}", path, e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()) == expected)
    })
    .await
    .map_err(|e| format!("Checksum task panicked: {}", e))?
}
