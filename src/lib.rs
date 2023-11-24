use rand::{seq::SliceRandom, Rng};

const SUDOKU_EXAMPLE: &str = "
    024007000
    600000000
    003680415
    431005000
    500000032
    790000060
    209710800
    040093000
    310004750
";

fn range(index: usize) -> std::ops::Range<usize> {
    index - index % 3..index + 3 - index % 3
}

fn init_temperature() -> f64 {
    const COUNT: u32 = 200;
    let errors: Vec<usize> = (0..COUNT)
        .map(|_| {
            let mut sample = SudokuTable::default();
            sample.fill_random();
            sample.errors()
        })
        .collect();
    let mean = errors.iter().sum::<usize>() as f64 / f64::from(COUNT);
    let standard_deviation = errors
        .iter()
        .map(|&error| {
            let diff = mean - error as f64;
            diff * diff
        })
        .sum::<f64>()
        / f64::from(COUNT);
    standard_deviation
}

#[derive(Default, PartialEq, Copy, Clone)]
enum SudokuCell {
    Fixed(u32),
    Proposed(u32),
    #[default]
    Empty,
}

impl std::fmt::Display for SudokuCell {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                SudokuCell::Fixed(digit) | SudokuCell::Proposed(digit) =>
                    char::from_digit(digit, 10).unwrap(),
                SudokuCell::Empty => '·',
            }
        )
    }
}

impl TryFrom<char> for SudokuCell {
    type Error = String;
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_digit(10) {
            Some(0) => Ok(SudokuCell::Empty),
            Some(digit) => Ok(SudokuCell::Fixed(digit)),
            None => Err(format!("Incorrect cell initialization value {value}")),
        }
    }
}

impl SudokuCell {
    fn filled_with(&self, value: u32) -> bool {
        matches!(self, SudokuCell::Proposed(inner_value) | SudokuCell::Fixed(inner_value) if *inner_value == value)
    }
}

enum Direction {
    Backward,
    Forward,
}

#[derive(Default, Clone)]
struct SudokuTable {
    table: [[SudokuCell; 9]; 9],
    errors: usize,
}

impl std::fmt::Display for SudokuTable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "┌───┬───┬───┐")?;
        for i in 0..9 {
            if i > 0 && i % 3 == 0 {
                writeln!(f, "├───┼───┼───┤")?;
            }
            for j in 0..9 {
                if j % 3 == 0 {
                    write!(f, "│")?;
                }
                write!(f, "{}", self.table[i][j])?;
            }
            writeln!(f, "│")?;
        }
        writeln!(f, "└───┴───┴───┘")?;
        Ok(())
    }
}

impl TryFrom<&str> for SudokuTable {
    type Error = String;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut table: [[SudokuCell; 9]; 9] = Default::default();
        for (row, line) in value
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(9)
            .enumerate()
        {
            for (column, cell) in line.chars().map(SudokuCell::try_from).take(9).enumerate() {
                table[row][column] = cell?;
            }
        }
        let mut result = SudokuTable { table, errors: 0 };
        result.calculate_errors();
        Ok(result)
    }
}

impl SudokuTable {
    fn fill_random(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..9 {
            for j in 0..9 {
                if self.table[i][j] == SudokuCell::Empty {
                    loop {
                        let new_value = rng.gen_range(1..10);
                        if self.value_count_in_square(i, j, new_value) == 0 {
                            self.table[i][j] = SudokuCell::Proposed(new_value);
                            break;
                        }
                    }
                }
            }
        }
        self.calculate_errors();
    }

    fn value_count_in_square(&self, row: usize, column: usize, value: u32) -> usize {
        let mut sum = 0;
        for i in range(row) {
            for j in range(column) {
                if self.table[i][j].filled_with(value) {
                    sum += 1;
                }
            }
        }
        sum
    }

    fn value_count_in_row(&self, row: usize, value: u32) -> usize {
        let mut sum = 0;
        for j in 0..9 {
            if self.table[row][j].filled_with(value) {
                sum += 1;
            }
        }
        sum
    }

    fn value_count_in_column(&self, column: usize, value: u32) -> usize {
        let mut sum = 0;
        for i in 0..9 {
            if self.table[i][column].filled_with(value) {
                sum += 1;
            }
        }
        sum
    }

    fn calculate_errors(&mut self) {
        self.errors = 0;
        for value in 1..=9 {
            for i in 0..9 {
                let count = self.value_count_in_row(i, value);
                if count > 1 {
                    self.errors += count - 1;
                }
            }
            for j in 0..9 {
                let count = self.value_count_in_column(j, value);
                if count > 1 {
                    self.errors += count - 1;
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    let count = self.value_count_in_square(i * 3, j * 3, value);
                    if count > 1 {
                        self.errors += count - 1;
                    }
                }
            }
        }
    }

    fn errors(&self) -> usize {
        self.errors
    }

    fn acceptance_probability(&self, previous_errors: usize, temperature: f64) -> f64 {
        f64::exp(-(self.errors as f64 - previous_errors as f64) / temperature)
    }

    fn swap_cells(&mut self, i1: usize, j1: usize, i2: usize, j2: usize) {
        let temp = self.table[i1][j1];
        let temp = std::mem::replace(&mut self.table[i2][j2], temp);
        let _ = std::mem::replace(&mut self.table[i1][j1], temp);
    }

    fn propose_change(&self) -> SudokuTable {
        if self.errors == 0 {
            // There is no sence to break current solution
            return self.clone();
        }
        let mut rng = rand::thread_rng();
        loop {
            // Choose 3x3 square for changing
            let (square_i, square_j) = (rng.gen_range(0..3), rng.gen_range(0..3));
            // Get all proposed cells
            let proposed = (0..3)
                .flat_map(|i| {
                    (0..3).filter_map(move |j| {
                        match self.table[square_i * 3 + i][square_j * 3 + j] {
                            SudokuCell::Proposed(_) => Some((i, j)),
                            _ => None,
                        }
                    })
                })
                .collect::<Vec<_>>();
            if proposed.len() < 2 {
                // there is no cells to swap in this square
                continue;
            }
            // println!("{proposed:?}");
            let selected = proposed.choose_multiple(&mut rng, 2).collect::<Vec<_>>();
            let mut changed_table = self.clone();
            changed_table.swap_cells(selected[0].0, selected[0].1, selected[1].0, selected[1].1);
            changed_table.calculate_errors();

            break changed_table;
        }
    }

    /// Decision for every step in brute force
    fn make_decision(
        &mut self,
        i: usize,
        j: usize,
        starting_number: u32,
    ) -> (SudokuCell, Direction) {
        for n in starting_number..=9 {
            if self.value_count_in_row(i, n) == 0
                && self.value_count_in_column(j, n) == 0
                && self.value_count_in_square(i, j, n) == 0
            {
                return (SudokuCell::Proposed(n), Direction::Forward);
            }
        }
        // In case of overflow go back
        (SudokuCell::Empty, Direction::Backward)
    }

    /// Solvers
    fn simulated_annealing(&mut self, mut temperature: f64, samples_per_temperature: usize) {
        let mut rng = rand::thread_rng();
        while self.errors > 0 {
            let errors = self.errors;
            // don't know peak random or best sample, both doesn't give final solution
            // let sample_index = rng.gen_range(0..samples_per_temperature);
            let sample = (0..)
                .map(|_| {
                    let new_table = self.propose_change();
                    println!(
                        "from {} to {} got {}",
                        self.errors,
                        new_table.errors(),
                        new_table.acceptance_probability(errors, temperature)
                    );
                    new_table
                })
                .filter(|sample| {
                    rng.gen_range(0.0..1.0) < sample.acceptance_probability(errors, temperature)
                })
                .take(samples_per_temperature)
                .max_by_key(|sample| sample.errors());
            // .take(sample_index + 1)
            // .last();
            if let Some(sample) = sample {
                let _ = std::mem::replace(self, sample);
                temperature *= 0.99;
            }
        }
    }

    fn optimized_brute_force(&mut self) -> bool {
        // clear all proposed values
        for i in 0..9 {
            for j in 0..9 {
                if let SudokuCell::Proposed(_) = self.table[i][j] {
                    self.table[i][j] = SudokuCell::Empty;
                }
            }
        }
        // Init starting state
        let (mut i, mut j) = (0, 0);
        let mut direction = Direction::Forward;
        loop {
            //// Here code for visual animation of algorithm
            // println!("{self}");
            // std::thread::sleep(std::time::Duration::from_millis(40));
            // change current cell and choose direction
            match self.table[i][j] {
                SudokuCell::Fixed(_) => {}
                SudokuCell::Proposed(9) => {
                    direction = Direction::Backward;
                    self.table[i][j] = SudokuCell::Empty;
                }
                SudokuCell::Proposed(number) => {
                    (self.table[i][j], direction) = self.make_decision(i, j, number + 1);
                }
                SudokuCell::Empty => {
                    (self.table[i][j], direction) = self.make_decision(i, j, 1);
                }
            }
            // move to next cell
            match direction {
                Direction::Forward => {
                    j += 1;
                    if j == 9 {
                        j = 0;
                        i += 1;
                        if i == 9 {
                            self.calculate_errors();
                            return true;
                        }
                    }
                }
                Direction::Backward => {
                    if j > 0 {
                        j -= 1;
                    } else {
                        j = 8;
                        if i > 0 {
                            i -= 1;
                        } else {
                            // Looks like there is no valid solution
                            return false;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brute_force_works() {
        let mut sudoku = SudokuTable::try_from(SUDOKU_EXAMPLE).unwrap();
        sudoku.calculate_errors();
        println!("{sudoku}");
        println!("{}", sudoku.errors());
        assert!(sudoku.optimized_brute_force());
        println!("{sudoku}");
        assert_eq!(sudoku.errors(), 0);

        //// Here attempt to do SA with proper initialization
        // sudoku.fill_random();
        // println!("{sudoku}");
        // let starting_temperature = init_temperature();
        // sudoku.simulated_annealing(starting_temperature, 100);
    }
}
