// 初始化 particles.js
particlesJS.load('particles-js', 'static/particles.json', function () {
    console.log('particles.js loaded - callback');
});

// 拖放文件支持
const dropZone = document.querySelector('.upload-box');
dropZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropZone.style.borderColor = 'blue';
});
dropZone.addEventListener('dragleave', function (e) {
    dropZone.style.borderColor = '#ccc';
});
dropZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropZone.style.borderColor = '#ccc';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const fileName = fileInput.files[0] ? fileInput.files[0].name : '';
    const label = document.querySelector('label[for="fileInput"]');
    label.innerHTML = `<h3>已选择文件：${fileName}</h3><p>点击开始识别</p>`;

    const audioPlayer = document.getElementById('audioPlayer');
    const file = fileInput.files[0];
    if (file) {
        const objectURL = URL.createObjectURL(file);
        audioPlayer.src = objectURL;
        audioPlayer.style.display = 'block';
    }
}

document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.upload.addEventListener('progress', function (e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            document.getElementById('progressBar').style.width = percentComplete + '%';
        }
    });
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                const resultDiv = document.getElementById('result');
                if (data.error) {
                    resultDiv.innerHTML = `<div style="color: red;">错误：${data.error}</div>`;
                } else {
                    let probsHtml = '<div style="color: gray;">概率分布：</div>';
                    data.probabilities.forEach(item => {
                        probsHtml += `<div style="color: gray;">${item.genre}: ${item.probability}</div>`;
                    });

                    let recommendedSongsHtml = '<table><tr><th>标题</th><th>时长</th><th>艺术家</th><th>播放</th></tr>';
                    data.recommended_songs.forEach(song => {
                        recommendedSongsHtml += `<tr><td>${song.title}</td><td>${song.duration}</td><td>${song.artist}</td><td><audio controls><source src="${song.audio_path}" type="audio/mpeg"></audio></td></tr>`;
                    });
                    recommendedSongsHtml += '</table>';

                    resultDiv.innerHTML = `
                                    <div class="result-container">
                                        <div style="color: black; font-size: 1.2em;">
                                            预测结果：${data.genre}
                                        </div>
                                        ${probsHtml}
                                    </div>
                                    <div class="recommended-container">
                                        <div>相似歌曲推荐：</div>
                                        ${recommendedSongsHtml}
                                    </div>
                                `;
                }
            } else {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `<div style="color: red;">错误：请求失败，状态码 ${xhr.status}</div>`;
            }
        }
    };
    xhr.send(formData);
});