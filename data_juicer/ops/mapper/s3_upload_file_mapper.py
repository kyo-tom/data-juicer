import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper

OP_NAME = "s3_upload_file_mapper"


@OPERATORS.register_module(OP_NAME)
class S3UploadFileMapper(Mapper):
    """Mapper to upload local files to S3 and update paths to S3 URLs.

    This operator uploads files from local paths to S3 storage. It supports:
    - Uploading multiple files concurrently
    - Updating file paths in the dataset to S3 URLs
    - Optional deletion of local files after successful upload
    - Custom S3 endpoints (for S3-compatible services like MinIO)
    - Skipping already uploaded files (based on S3 key)

    The operator processes nested lists of paths, maintaining the original structure
    in the output."""

    _batched_op = True

    def __init__(
        self,
        upload_field: str = None,
        s3_bucket: str = None,
        s3_prefix: str = '',
        # S3 credentials
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        aws_region: str = None,
        endpoint_url: str = None,
        # Upload options
        remove_local: bool = False,
        skip_existing: bool = True,
        max_concurrent: int = 10,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param upload_field: The field name containing file paths to upload.
        :param s3_bucket: S3 bucket name to upload files to.
        :param s3_prefix: Prefix (folder path) in S3 bucket. E.g., 'videos/' or 'data/videos/'.
        :param aws_access_key_id: AWS access key ID for S3.
        :param aws_secret_access_key: AWS secret access key for S3.
        :param aws_session_token: AWS session token for S3 (optional).
        :param aws_region: AWS region for S3.
        :param endpoint_url: Custom S3 endpoint URL (for S3-compatible services).
        :param remove_local: Whether to delete local files after successful upload.
        :param skip_existing: Whether to skip uploading if file already exists in S3.
        :param max_concurrent: Maximum concurrent uploads.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.upload_field = upload_field
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix and not s3_prefix.endswith('/') else s3_prefix or ''
        self.remove_local = remove_local
        self.skip_existing = skip_existing
        self.max_concurrent = max_concurrent

        if not self.s3_bucket:
            raise ValueError("s3_bucket must be specified")

        if not (aws_access_key_id and aws_secret_access_key):
            raise ValueError("AWS credentials (aws_access_key_id and aws_secret_access_key) must be provided")

        # Store S3 configuration (don't create client here to avoid serialization issues)
        self.s3_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        }
        if aws_session_token:
            self.s3_config['aws_session_token'] = aws_session_token
        if aws_region:
            self.s3_config['region_name'] = aws_region
        if endpoint_url:
            self.s3_config['endpoint_url'] = endpoint_url

        self._s3_client = None
        logger.info(f"S3 upload mapper initialized: bucket={s3_bucket}, prefix={self.s3_prefix}, endpoint={endpoint_url or 'default'}")

    @property
    def s3_client(self):
        """Lazy initialization of S3 client to avoid serialization issues with Ray."""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', **self.s3_config)
            logger.debug("S3 client initialized (lazy)")
        return self._s3_client

    def _is_s3_url(self, path: str) -> bool:
        """Check if the path is already an S3 URL."""
        return isinstance(path, str) and path.startswith('s3://')

    def _check_s3_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def _upload_to_s3(self, local_path: str) -> tuple:
        """Upload a single file to S3.

        :param local_path: Local file path to upload
        :return: (status, s3_url, error_message)
        """
        # Already an S3 URL, skip
        if self._is_s3_url(local_path):
            logger.debug(f"Path is already S3 URL: {local_path}")
            return 'skipped', local_path, None

        # Check if file exists locally
        if not os.path.exists(local_path):
            error_msg = f"Local file not found: {local_path}"
            logger.warning(error_msg)
            return 'failed', local_path, error_msg

        try:
            # Construct S3 key
            filename = os.path.basename(local_path)
            s3_key = self.s3_prefix + filename
            s3_url = f's3://{self.s3_bucket}/{s3_key}'

            # Check if file already exists in S3
            if self.skip_existing and self._check_s3_exists(s3_key):
                logger.debug(f"File already exists in S3, skipping: {s3_url}")

                # Delete local file if configured
                if self.remove_local:
                    try:
                        os.remove(local_path)
                        logger.debug(f"Removed local file: {local_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove local file {local_path}: {e}")

                return 'exists', s3_url, None

            # Upload to S3
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            logger.info(f"Uploaded: {local_path} -> {s3_url}")

            # Delete local file if configured
            if self.remove_local:
                try:
                    os.remove(local_path)
                    logger.debug(f"Removed local file: {local_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove local file {local_path}: {e}")

            return 'success', s3_url, None

        except ClientError as e:
            error_msg = f"S3 upload failed: {e}"
            logger.error(error_msg)
            return 'failed', local_path, error_msg
        except Exception as e:
            error_msg = f"Upload error: {e}"
            logger.error(error_msg)
            return 'failed', local_path, error_msg

    def upload_files_concurrent(self, paths: List[str]) -> List[tuple]:
        """Upload multiple files concurrently.

        :param paths: List of local file paths
        :return: List of (idx, status, s3_url, error_message) tuples
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {executor.submit(self._upload_to_s3, path): idx for idx, path in enumerate(paths)}

            for future in futures:
                idx = futures[future]
                try:
                    status, s3_url, error = future.result()
                    results.append((idx, status, s3_url, error))
                except Exception as e:
                    error_msg = f"Upload thread error: {e}"
                    logger.error(error_msg)
                    results.append((idx, 'failed', paths[idx], error_msg))

        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        return results

    def _flat_paths(self, nested_paths):
        """Flatten nested paths while preserving structure information."""
        flat_paths = []
        structure_info = []  # (original_index, sub_index)

        for idx, paths in enumerate(nested_paths):
            if isinstance(paths, list):
                for sub_idx, path in enumerate(paths):
                    flat_paths.append(path)
                    structure_info.append((idx, sub_idx))
            else:
                flat_paths.append(paths)
                structure_info.append((idx, -1))  # -1 means single element

        return flat_paths, structure_info

    def _create_path_struct(self, nested_paths) -> List:
        """Create path structure for output."""
        reconstructed = []
        for item in nested_paths:
            if isinstance(item, list):
                reconstructed.append([None] * len(item))
            else:
                reconstructed.append(None)
        return reconstructed

    def upload_nested_paths(self, nested_paths: List[Union[str, List[str]]]):
        """Upload nested paths with structure preservation.

        :param nested_paths: Nested list of file paths
        :return: (reconstructed_paths, failed_info)
        """
        flat_paths, structure_info = self._flat_paths(nested_paths)

        # Upload all files concurrently
        upload_results = self.upload_files_concurrent(flat_paths)

        # Reconstruct nested structure
        reconstructed_paths = self._create_path_struct(nested_paths)

        failed_info = ""
        success_count = 0
        failed_count = 0
        skipped_count = 0
        exists_count = 0

        for i, (idx, status, s3_url, error) in enumerate(upload_results):
            orig_idx, sub_idx = structure_info[i]

            if status == 'success':
                success_count += 1
            elif status == 'failed':
                failed_count += 1
                if error:
                    failed_info += f"\n{flat_paths[i]}: {error}"
            elif status == 'skipped':
                skipped_count += 1
            elif status == 'exists':
                exists_count += 1

            # Update path in reconstructed structure
            if sub_idx == -1:
                reconstructed_paths[orig_idx] = s3_url
            else:
                reconstructed_paths[orig_idx][sub_idx] = s3_url

        # Log summary
        logger.info(
            f"Upload summary: {success_count} uploaded, {exists_count} already exists, "
            f"{skipped_count} skipped, {failed_count} failed"
        )

        return reconstructed_paths, failed_info

    def process_batched(self, samples):
        """Process a batch of samples."""
        if self.upload_field not in samples or not samples[self.upload_field]:
            return samples

        batch_nested_paths = samples[self.upload_field]

        # Upload files and get S3 URLs
        reconstructed_paths, failed_info = self.upload_nested_paths(batch_nested_paths)

        # Update the field with S3 URLs
        samples[self.upload_field] = reconstructed_paths

        if len(failed_info):
            logger.error(f"Failed uploads:\n{failed_info}")

        return samples
