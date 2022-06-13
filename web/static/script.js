// todo: pass a parameter in the function such that only given menu gets activated
const initNavBar = () => {
    const menusEl = document.querySelectorAll('.side-bar ul li')
    menusEl.forEach(menu => menu.addEventListener('click', () => {
        const menuActiveEl = document.querySelector('.side-bar ul li.active')
        menuActiveEl.classList.remove('active')
        menu.classList.add('active')
    }))
}
let input = document.getElementById("inputTag");
let imageName = document.getElementById("imageName")

input.addEventListener("change", () => {
    let inputImage = document.querySelector("input[type=file]").files[0];
    imageName.innerText = inputImage.name;
})
initNavBar()

function checkFiles(files) {
    if (files.length > 10) {
        alert("length exceeded; files have been truncated");

        let list = new DataTransfer;
        for (let i = 0; i < 10; i++)
            list.items.add(files[i])

        document.getElementById('files').files = list.files
    }
}
// input should not be empty
function validateForm() {
    let files = document.getElementById('inputTag');
    if (files.value === "") {
        alert("Please select a file");
        return false;
    }
}

function readURL(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            $('#blah')
                .attr('src', e.target.result)
                .width(150)
                .height(200);
        };
        reader.readAsDataURL(input.files[0]);
    }
}